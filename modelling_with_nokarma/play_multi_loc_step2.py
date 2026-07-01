# STEP 2: two attackers (Loc 9 + Loc 16)  in the 16-location simulation.

# Absolute caps, Prop A.1 utility, discounted-cumulative over T quarters.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import (N_TYPES, TYPE_PROBS, round_assignment,
                        location_attacks, find_pure_nash, C_APPENDIX,
                        LocPredictor)
from play_multi_loc_attack import (build_full_model, LOCS_ALL,
                                   solve_lp_with_kappa)

# Planner used by the matching LP.
#   "strat" 
#   "learned" 
PLANNER = "learned"

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11.5,
})

BETA        = 0.9
N_SEEDS     = 20
ATTACKERS   = [9, 16]
ATTACK_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
HERE        = os.path.dirname(__file__)
BETA_GRID   = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# Quota tightness: minimum intake kappa_l = floor(PHI * alpha_l * arrivals).
# PHI = 0    -> there is no minimum only the maximum capacity constraint.
# PHI = 1.0  -> each location forced to take its full fair share = most defense.
PHI = 0
CACHE = os.path.join(HERE,
                     f"step2_counts_loc9_16_{PLANNER}_phi{int(PHI*100):02d}.npz")


def floor_vector_phi(phi, alpha, n_t):
    """kappa_l = floor(phi * alpha_l * n_t). Sum <= phi*n_t <= n_t -> feasible."""
    return np.floor(phi * alpha * n_t).astype(float)


def collect_assignments(model, idx9, idx16, n_seeds):
    arrivals  = model["quarterly_arrivals"]
    T         = len(arrivals)
    n_loc     = len(model["locs"])
    cost_mask = model["cost_mask"]
    theta     = model["theta"]
    cap_abs   = model["cap"]
    alpha     = model["alpha"]

    K = len(ATTACK_GRID)
    counts = np.zeros((K, K, n_seeds, T, n_loc, N_TYPES))
    emp    = np.zeros((K, K, n_seeds))     # mean employment per cell per seed

    for i, lam9 in enumerate(ATTACK_GRID):
        for j, lam16 in enumerate(ATTACK_GRID):
            atk_vec = np.zeros(n_loc)
            atk_vec[idx9]  = lam9
            atk_vec[idx16] = lam16
            for s in range(n_seeds):
                np.random.seed(s)
                if PLANNER == "learned":
                    predictor = LocPredictor(n_loc, random_state=s)
                    types_buf, loc_buf, out_buf = [], [], []
                emp_num = 0.0; emp_den = 0
                for t in range(T):
                    n_t = int(arrivals[t])
                    if n_t == 0:
                        continue
                    types  = np.random.choice(N_TYPES, size=n_t, p=TYPE_PROBS)
                    V_true = theta[types]
                    lambdas = location_attacks(types, atk_vec, cost_mask)
                    if PLANNER == "strat":
                        V_for_lp = (1.0 - lambdas) * V_true
                    else:                              # "learned"
                        V_for_lp = predictor.predict(types)
                    floor = floor_vector_phi(PHI, alpha, n_t)
                    x_frac = solve_lp_with_kappa(V_for_lp, cap_abs, floor)
                    a = round_assignment(x_frac).argmax(axis=1)
                    outcomes = np.array([
                        np.random.binomial(1, (1 - lambdas[ii, a[ii]])
                                              * V_true[ii, a[ii]])
                        for ii in range(n_t)
                    ])
                    emp_num += outcomes.sum(); emp_den += n_t
                    if PLANNER == "learned":
                        types_buf.append(types); loc_buf.append(a)
                        out_buf.append(outcomes)
                        predictor.train(np.concatenate(types_buf),
                                        np.concatenate(loc_buf),
                                        np.concatenate(out_buf))
                    for ii in range(n_t):
                        counts[i, j, s, t, a[ii], types[ii]] += 1
                emp[i, j, s] = emp_num / max(emp_den, 1)
    return counts, emp, T


def load_or_collect(model, idx9, idx16):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if int(d["n_seeds"]) == N_SEEDS and "emp" in d.files:
            return d["counts"], d["emp"], int(d["T"])
    counts, emp, T = collect_assignments(model, idx9, idx16, N_SEEDS)
    np.savez(CACHE, counts=counts, emp=emp, T=T, n_seeds=N_SEEDS)
    return counts, emp, T


def aggregate_payoff(counts, model, idx9, idx16, beta, c=C_APPENDIX):
    K, _, n_seeds, T, n_loc, n_types = counts.shape
    theta     = model["theta"]
    cost_mask = model["cost_mask"]
    p_lk      = theta.T                      # (n_loc, n_types)
    weights   = beta ** np.arange(T)

    U9  = np.zeros((K, K))
    U16 = np.zeros((K, K))
    denominator = 1.0 - beta + beta * p_lk   # (n_loc, n_types), cell-independent
    for i in range(K):
        for j in range(K):
            lam_vec = np.zeros(n_loc)
            lam_vec[idx9]  = ATTACK_GRID[i]
            lam_vec[idx16] = ATTACK_GRID[j]
            lam_eff = lam_vec[:, None] * cost_mask.T          # (n_loc, n_types)
            numerator = 1.0 + beta * lam_eff * p_lk
            u_indiv = -c * numerator / denominator
            u_round = np.einsum('stlk,lk->stl', counts[i, j], u_indiv)
            U_per_seed = (u_round * weights[None, :, None]).sum(axis=1)
            U_loc = U_per_seed.mean(axis=0)
            U9[i, j]  = U_loc[idx9]
            U16[i, j] = U_loc[idx16]
    return U9, U16


def plot_heatmap(U9, U16, nash):
    grid = ATTACK_GRID
    K = len(grid)
    tick = [f"{int(x*100)}%" for x in grid]
    vmin = min(U9.min(), U16.min())
    vmax = max(U9.max(), U16.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             gridspec_kw={"wspace": 0.30})
    panels = [(axes[0], U9, r"$U_{9}$  (Loc 9)"),
              (axes[1], U16, r"$U_{16}$  (Loc 16)")]
    last_im = None
    for ax, M, name in panels:
        im = ax.imshow(M, cmap="RdYlGn", origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_xticks(range(K)); ax.set_xticklabels(tick)
        ax.set_yticks(range(K)); ax.set_yticklabels(tick)
        ax.set_xlabel(r"$\lambda_{16}$")
        ax.set_ylabel(r"$\lambda_{9}$")
        ax.set_title(name)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=9, color="black")
        for (i, j) in nash:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=3.0, zorder=5))
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    out = os.path.join(HERE, f"step2_loc9_16_heatmap_{PLANNER}_phi{int(PHI*100):02d}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()



def plot_employment_heatmap(emp, nash):
    grid = ATTACK_GRID
    K = len(grid)
    tick = [f"{int(x*100)}%" for x in grid]
    E = emp.mean(axis=2)                  

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(E, cmap="RdYlGn", origin="lower", aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels(tick)
    ax.set_yticks(range(K)); ax.set_yticklabels(tick)
    ax.set_xlabel(r"$\lambda_{16}$")
    ax.set_ylabel(r"$\lambda_{9}$")
    ax.set_title("Total employment rate")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{E[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    for (i, j) in nash:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=3.0, zorder=5))
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    plt.tight_layout()
    out = os.path.join(HERE, f"step2_loc9_16_employment_{PLANNER}_phi{int(PHI*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


    e_truth = E[0, 0]
    if nash:
        i, j = nash[0]
        e_ne = E[i, j]
        print(f"\n  Total employment truthful (0,0): {e_truth:.4f}")
        print(f"  Total employment at NE ({grid[i]:.2f},{grid[j]:.2f}): "
              f"{e_ne:.4f}")
        print(f"  Social loss: {e_truth - e_ne:+.4f} absolute, "
              f"{100*(e_ne - e_truth)/e_truth:+.1f}% relative")


def plot_beta_sensitivity(counts, model, idx9, idx16):
    ne9 = []; ne16 = []
    for beta in BETA_GRID:
        U9, U16 = aggregate_payoff(counts, model, idx9, idx16, beta)
        nash = find_pure_nash(U9, U16)
        if nash:
            i, j = nash[0]
            ne9.append(ATTACK_GRID[i]); ne16.append(ATTACK_GRID[j])
        else:
            ne9.append(np.nan); ne16.append(np.nan)
        print(f"    beta={beta:.2f}  NE=({ne9[-1]}, {ne16[-1]})")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(BETA_GRID, ne9, marker="o", color="tab:blue", lw=2.4,
            markersize=9, markeredgecolor="white", markeredgewidth=1.0,
            label=r"$\lambda^\star_{9}$  (Loc 9)")
    ax.plot(BETA_GRID, np.array(ne16) + 0.02, marker="s", color="tab:orange",
            lw=2.4, markersize=9, markeredgecolor="white", markeredgewidth=1.0,
            linestyle="--", label=r"$\lambda^\star_{16}$  (Loc 16)")
    ax.set_xlabel(r"Discount factor $\beta$")
    ax.set_ylabel("Equilibrium attack level")
    ax.set_ylim(-0.12, 1.18)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", framealpha=0.95)
    ax.set_title("(Step 2)  Equilibrium attack level vs $\\beta$ "
                 "(Loc 9 + Loc 16 attack, 14 truthful)", loc="left")
    plt.tight_layout()
    out = os.path.join(HERE, f"step2_loc9_16_beta_sensitivity_{PLANNER}_phi{int(PHI*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print(f"STEP 2: two attackers = Loc 9 + Loc 16, "
          f"absolute caps, planner={PLANNER}, PHI={PHI}, beta = {BETA}")
    model = build_full_model(LOCS_ALL)
    idx9  = LOCS_ALL.index(9)
    idx16 = LOCS_ALL.index(16)

    counts, emp, T = load_or_collect(model, idx9, idx16)
    U9, U16 = aggregate_payoff(counts, model, idx9, idx16, BETA)
    nash = find_pure_nash(U9, U16)
    grid = ATTACK_GRID

    plot_heatmap(U9, U16, nash)
    plot_employment_heatmap(emp, nash)
    plot_beta_sensitivity(counts, model, idx9, idx16)


if __name__ == "__main__":
    main()
