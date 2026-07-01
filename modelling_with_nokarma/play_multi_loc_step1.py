
# Question: do the results from the 2-location case still hold when we use the
# full HIAS network?

# Setup:
#   - 16 HIAS locations (all except Loc 11, which has no capacity)
#   - Only one attacker (Loc 9); the other 15 locations stay truthful
#   - Same LP as before: capacity caps ceil(alpha_l * r), no minimum intake
#   - Same utility as before, beta=0.9

# Since only one location can attack, there is no longer a 5x5 grid.
# The attacker simply chooses lambda in {0, .25, .5, .75, 1}.

# We therefore look at:
#   - utility vs lambda (does attacking pay off?)
#   - employment vs lambda
#   - how the attack benefit changes for different beta values
#
#
# Output figures:
#   step1_loc9_utility_vs_lambda.png
#   step1_loc9_employment_vs_lambda.png
#   step1_loc9_beta_sensitivity.png

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import (N_TYPES, TYPE_PROBS, solve_lp, round_assignment,
                        location_attacks, C_APPENDIX, LocPredictor)
from play_multi_loc_attack import (build_full_model, LOCS_ALL,
                                   solve_lp_with_kappa)

# Planner used by the matching LP:
#   "strat"  -> planner knows every attack and uses V_strat = (1 - lambda) * V_true.
#   "learned" ->  Never sees lambda; only learns the success rates from data, with a learning lag.
PLANNER = "learned"

# Capacity regime:
#   "absolute" -> each location capped at its real HIAS quota (Loc 9 = 288). 
#   "proportional" -> each location capped at ceil(alpha_l * arrivals)
CAP_MODE = "absolute"


try:
    from ibr_common import TYPE_NAMES
except Exception:
    TYPE_NAMES = [f"type {k}" for k in range(N_TYPES)]

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11.5,
})

BETA        = 0.9
N_SEEDS     = 30
ATTACKER    = 9
ATTACK_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
HERE        = os.path.dirname(__file__)
BETA_GRID   = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# PHI controls the per-location MINIMUM intake (floor):
#     kappa_l = floor(PHI * alpha_l * arrivals)
# The HIAS upper caps (cap_abs) are ALWAYS active; PHI only adds a floor.
#   PHI = 0   -> no floor, only the upper caps constrain (Step-1 baseline).
#   PHI = 1   -> each location forced to take its full fair share = most defense.
PHI = 0


def floor_vector_phi(phi, alpha, n_t):
    """kappa_l = floor(phi * alpha_l * n_t). Sum <= phi*n_t <= n_t -> feasible."""
    return np.floor(phi * alpha * n_t).astype(float)


def simulate(model, attacker_idx, lam, n_seeds):
    arrivals  = model["quarterly_arrivals"]
    T         = len(arrivals)
    n_loc     = len(model["locs"])
    cost_mask = model["cost_mask"]
    alpha     = model["alpha"]
    theta     = model["theta"]

    atk_vec = np.zeros(n_loc)
    atk_vec[attacker_idx] = lam

    counts        = np.zeros((n_seeds, T, n_loc, N_TYPES))
    emp_round     = np.zeros((n_seeds, T))
    pertype_emp_s = np.zeros((n_seeds, N_TYPES))   # mean outcome per type
    pertype_n_s   = np.zeros((n_seeds, N_TYPES))

    for s in range(n_seeds):
        np.random.seed(s)
        emp_num = 0.0; emp_den = 0
        pt_num = np.zeros(N_TYPES); pt_den = np.zeros(N_TYPES)
        if PLANNER == "learned":
            predictor = LocPredictor(n_loc, random_state=s)
            types_buf, loc_buf, out_buf = [], [], []
        for t in range(T):
            n_t = int(arrivals[t])
            if n_t == 0:
                continue
            types  = np.random.choice(N_TYPES, size=n_t, p=TYPE_PROBS)
            V_true = theta[types]
            lambdas = location_attacks(types, atk_vec, cost_mask)
            if PLANNER == "strat":
                V_for_lp = (1.0 - lambdas) * V_true
            elif PLANNER == "learned":
                V_for_lp = predictor.predict(types)
            else:
                raise ValueError(f"Unknown PLANNER={PLANNER!r}")
            if CAP_MODE == "absolute":
                floor = floor_vector_phi(PHI, alpha, n_t)
                x_frac = solve_lp_with_kappa(V_for_lp, model["cap"], floor)
                x = round_assignment(x_frac)
            else:
                x = round_assignment(solve_lp(V_for_lp, alpha))
            assigned = x.argmax(axis=1)
            outcomes = np.array([
                np.random.binomial(1, (1 - lambdas[i, assigned[i]])
                                      * V_true[i, assigned[i]])
                for i in range(n_t)
            ])
            if PLANNER == "learned":
                types_buf.append(types); loc_buf.append(assigned)
                out_buf.append(outcomes)
                predictor.train(np.concatenate(types_buf),
                                np.concatenate(loc_buf),
                                np.concatenate(out_buf))
            for i in range(n_t):
                counts[s, t, assigned[i], types[i]] += 1
            emp_round[s, t] = outcomes.mean()
            emp_num += outcomes.sum(); emp_den += n_t
            for k in range(N_TYPES):
                m = (types == k)
                pt_num[k] += outcomes[m].sum(); pt_den[k] += m.sum()
        pertype_emp_s[s] = np.where(pt_den > 0, pt_num / np.maximum(pt_den, 1),
                                     np.nan)
        pertype_n_s[s]   = pt_den

    return counts, emp_round, pertype_emp_s



def prop_a1_discounted_utility(counts, model, attacker_idx, lam, beta,
                            c=C_APPENDIX):
    n_seeds, T, n_loc, n_types = counts.shape
    theta     = model["theta"]
    cost_mask = model["cost_mask"]

    atk_vec = np.zeros(n_loc)
    atk_vec[attacker_idx] = lam

    # effective per-(loc, type) attack level (0 outside the cost-mask)
    lam_eff = np.zeros((n_loc, n_types))
    for l in range(n_loc):
        for k in range(n_types):
            lam_eff[l, k] = atk_vec[l] * (1.0 if cost_mask[k, l] else 0.0)

    p_lk = theta.T                                  # (n_loc, n_types)
    numerator   = 1.0 + beta * lam_eff * p_lk
    denominator = 1.0 - beta + beta * p_lk
    u_indiv = -c * numerator / denominator          # (n_loc, n_types)

    # per (seed, round, loc) stage cost
    u_round = np.einsum('stlk,lk->stl', counts, u_indiv)
    weights = beta ** np.arange(T)
    U_per_seed = (u_round * weights[None, :, None]).sum(axis=1)  # (n_seeds, n_loc)
    return U_per_seed.mean(axis=0)                               # (n_loc,)



def main():
    print("=" * 78)
    print(f"STEP 1: 16-loc HIAS, single attacker = Loc {ATTACKER}, "
          f"{CAP_MODE} caps, planner={PLANNER}, PHI={PHI}, beta = {BETA}")
    print("=" * 78)
    model = build_full_model(LOCS_ALL)
    locs  = LOCS_ALL
    atk_idx = locs.index(ATTACKER)
    print(f"  Attacker Loc {ATTACKER}: cap={int(model['cap'][atk_idx])}, "
          f"alpha={model['alpha'][atk_idx]:.3f}")
    print(f"  Proportional cap at mean arrivals "
          f"({model['quarterly_arrivals'].mean():.0f}): "
          f"~{np.ceil(model['alpha'][atk_idx] * model['quarterly_arrivals'].mean()):.0f}/quarter")

    # store per-lambda results
    U_attacker   = np.zeros(len(ATTACK_GRID))
    U_others_sum = np.zeros(len(ATTACK_GRID))
    emp_total    = np.zeros(len(ATTACK_GRID))
    emp_total_sd = np.zeros(len(ATTACK_GRID))
    pertype_emp  = np.zeros((len(ATTACK_GRID), N_TYPES))
    counts_cache = {}

    print(f"\n  {'lambda':>7} | {'U_Loc9':>10} | {'sum U_others':>13} | "
          f"{'emp_total':>10}")
    print("  " + "-" * 50)
    for li, lam in enumerate(ATTACK_GRID):
        counts, emp_round, pertype_emp_s = simulate(model, atk_idx, lam,
                                                     N_SEEDS)
        counts_cache[lam] = counts
        U_all = prop_a1_discounted_utility(counts, model, atk_idx, lam, BETA)
        U_attacker[li]   = U_all[atk_idx]
        U_others_sum[li] = U_all.sum() - U_all[atk_idx]
        emp_total[li]    = emp_round.mean()
        emp_total_sd[li] = emp_round.mean(axis=1).std()
        pertype_emp[li]  = np.nanmean(pertype_emp_s, axis=0)
        print(f"  {lam:7.2f} | {U_attacker[li]:+10.2f} | "
              f"{U_others_sum[li]:+13.2f} | {emp_total[li]:10.4f}")

    
    plot_utility_curve(U_attacker, U_others_sum)

    plot_employment(emp_total, emp_total_sd, pertype_emp)

    plot_beta_sensitivity(model, atk_idx, counts_cache)


    best_li = int(np.argmax(U_attacker))
    print(f"\n  Attacker's best lambda (max U_Loc9): "
          f"{ATTACK_GRID[best_li]:.2f}")
    print(f"  Attack premium U(1)-U(0): "
          f"{U_attacker[-1] - U_attacker[0]:+.2f}")
    print(f"  Employment truthful->full attack: "
          f"{emp_total[0]:.4f} -> {emp_total[-1]:.4f} "
          f"({100*(emp_total[-1]-emp_total[0])/emp_total[0]:+.1f}%)")


def plot_utility_curve(U_attacker, U_others_sum):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(ATTACK_GRID, U_attacker, marker="o", color="tab:blue", lw=2.6,
            markersize=10, markeredgecolor="white", markeredgewidth=1.2,
            label=f"$U_{{{ATTACKER}}}$  (attacker)")
    # best response marker
    best_li = int(np.argmax(U_attacker))
    ax.scatter([ATTACK_GRID[best_li]], [U_attacker[best_li]], s=320,
               facecolors="none", edgecolors="tab:red", lw=2.5, zorder=6,
               label=f"best response $\\lambda^\\star = "
                     f"{ATTACK_GRID[best_li]:.2f}$")
    ax.set_xlabel(r"Attack level $\lambda$")
    ax.set_ylabel("Expected discounted utility")
    ax.set_xticks(ATTACK_GRID)
    ax.grid(alpha=0.3)
    ax.set_title(f"(Step 1)  Single-attacker utility vs attack level\n"
                 f"16-loc HIAS, Loc {ATTACKER} attacks, 15 truthful",
                 loc="left")
    ax.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(HERE, f"step1_loc9_utility_vs_lambda_{PLANNER}_phi{int(PHI*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")


def plot_employment(emp_total, emp_total_sd, pertype_emp):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 9.0), sharex=True,
                                  gridspec_kw={"height_ratios": [1.5, 1.0],
                                               "hspace": 0.12})
    # per-type
    cmap = plt.cm.tab10
    drop = pertype_emp[0] - pertype_emp[-1]
    worst = int(np.nanargmax(drop))
    for k in range(N_TYPES):
        line = pertype_emp[:, k]
        if np.all(np.isnan(line)):
            continue
        is_worst = (k == worst)
        ax.plot(ATTACK_GRID, line, marker="o",
                color=cmap(k), lw=3.0 if is_worst else 1.8,
                markersize=9 if is_worst else 6,
                alpha=1.0 if is_worst else 0.8,
                markeredgecolor="white", markeredgewidth=1.0,
                label=f"type {k}: {TYPE_NAMES[k]}"
                      f"{'  WORST' if is_worst else ''}")
    ax.set_ylabel("Employment rate by type")
    ax.grid(alpha=0.3)
    ax.set_title(f"(Step 1)  Employment vs attack level "
                 f"(Loc {ATTACKER} attacks, 15 truthful)", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3,
              framealpha=1.0, fontsize=9.5)

    # aggregate
    ax2.errorbar(ATTACK_GRID, emp_total, yerr=emp_total_sd,
                 marker="D", color="black", lw=3.0, markersize=12,
                 capsize=6, elinewidth=2.0, markeredgecolor="white",
                 markeredgewidth=1.8,
                 label="TOTAL employment")
    ax2.axhline(emp_total[0], color="#1f77b4", ls="--", lw=2.0, alpha=0.7,
                label="truthful baseline")
    ax2.set_xlabel(r"Attack level $\lambda$")
    ax2.set_ylabel("Total employment rate")
    ax2.set_xticks(ATTACK_GRID)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(HERE, f"step1_loc9_employment_vs_lambda_{PLANNER}_phi{int(PHI*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_beta_sensitivity(model, atk_idx, counts_cache):
    """Attack premium U(lambda=1) - U(lambda=0) for the attacker, vs beta."""
    counts0 = counts_cache[0.0]
    counts1 = counts_cache[1.0]
    premium = np.zeros(len(BETA_GRID))
    for bi, beta in enumerate(BETA_GRID):
        U0 = prop_a1_discounted_utility(counts0, model, atk_idx, 0.0, beta)
        U1 = prop_a1_discounted_utility(counts1, model, atk_idx, 1.0, beta)
        premium[bi] = U1[atk_idx] - U0[atk_idx]
        print(f"    beta={beta:.2f}  U_Loc9 premium U(1)-U(0) = "
              f"{premium[bi]:+.2f}")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(BETA_GRID, premium, marker="o", color="tab:blue", lw=2.6,
            markersize=9, markeredgecolor="white", markeredgewidth=1.1)
    ax.axhline(0, color="gray", lw=1.0)
    ax.fill_between(BETA_GRID, 0, premium, where=(premium >= 0),
                    color="tab:red", alpha=0.15)
    ax.fill_between(BETA_GRID, 0, premium, where=(premium < 0),
                    color="tab:green", alpha=0.15)
    ax.set_xlabel(r"Discount factor $\beta$")
    ax.set_ylabel("Utility gain from attacking")
    ax.set_title(f"(Step 1)  Attack incentive vs $\\beta$ "
                 f"(Loc {ATTACKER} attacks, 15 truthful)", loc="left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, f"step1_loc9_beta_sensitivity_{PLANNER}_phi{int(PHI*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
