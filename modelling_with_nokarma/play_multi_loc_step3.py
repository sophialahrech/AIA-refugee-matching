# STEP 3: minimum-intake floor as a defense lever.
#
# Each location must take at least kappa_l per quarter. The room to attack is
# the GAP between this floor and the location's fair share. 
# Expectation: incentive -> 0 as phi -> 1.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import (N_TYPES, TYPE_PROBS, round_assignment,
                        location_attacks, C_APPENDIX, LocPredictor)
from play_multi_loc_attack import (build_full_model, LOCS_ALL,
                                   solve_lp_with_kappa)

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11.5,
})

BETA      = 0.9
N_SEEDS   = 100
PLANNER   = "learned"
PHI_GRID  = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
BETA_GRID = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
HERE      = os.path.dirname(__file__)
CACHE     = os.path.join(HERE, f"step3_floor_{PLANNER}_v2.npz")


PROFILES = [(0.0, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
            (0.0, 0.75), (0.75, 0.75)]


def floor_vector_phi(phi, alpha, n_t):
    """kappa_l = floor(phi * alpha_l * n_t). Sum <= phi*n_t <= n_t -> feasible."""
    return np.floor(phi * alpha * n_t).astype(float)


def simulate_profile(model, lam9, lam16, phi, n_seeds):
    """Run T quarters under the learned planner with a phi quota floor."""
    arrivals  = model["quarterly_arrivals"]
    T         = len(arrivals)
    n_loc     = len(model["locs"])
    cost_mask = model["cost_mask"]
    theta     = model["theta"]
    cap_abs   = model["cap"]
    alpha     = model["alpha"]
    idx9      = model["locs"].index(9)
    idx16     = model["locs"].index(16)

    atk_vec = np.zeros(n_loc)
    atk_vec[idx9]  = lam9
    atk_vec[idx16] = lam16

    counts = np.zeros((n_seeds, T, n_loc, N_TYPES))
    emp_num = 0.0; emp_den = 0
    for s in range(n_seeds):
        np.random.seed(s)
        predictor = LocPredictor(n_loc, random_state=s)
        types_buf, loc_buf, out_buf = [], [], []
        for t in range(T):
            n_t = int(arrivals[t])
            if n_t == 0:
                continue
            types  = np.random.choice(N_TYPES, size=n_t, p=TYPE_PROBS)
            V_true = theta[types]
            lambdas = location_attacks(types, atk_vec, cost_mask)
            V_for_lp = predictor.predict(types)
            floor = floor_vector_phi(phi, alpha, n_t)
            x_frac = solve_lp_with_kappa(V_for_lp, cap_abs, floor)
            a = round_assignment(x_frac).argmax(axis=1)
            outcomes = np.array([
                np.random.binomial(1, (1 - lambdas[ii, a[ii]])
                                      * V_true[ii, a[ii]])
                for ii in range(n_t)
            ])
            emp_num += outcomes.sum(); emp_den += n_t
            types_buf.append(types); loc_buf.append(a); out_buf.append(outcomes)
            predictor.train(np.concatenate(types_buf),
                            np.concatenate(loc_buf),
                            np.concatenate(out_buf))
            for ii in range(n_t):
                counts[s, t, a[ii], types[ii]] += 1
    return counts, emp_num / max(emp_den, 1)


def prop_a1_utility(counts, model, lam9, lam16, beta, c=C_APPENDIX):
    """discounted-cumulative utility per location."""
    n_seeds, T, n_loc, n_types = counts.shape
    theta     = model["theta"]
    cost_mask = model["cost_mask"]
    idx9      = model["locs"].index(9)
    idx16     = model["locs"].index(16)
    lam_vec = np.zeros(n_loc)
    lam_vec[idx9]  = lam9
    lam_vec[idx16] = lam16
    p_lk = theta.T
    lam_eff = lam_vec[:, None] * cost_mask.T
    numerator   = 1.0 + beta * lam_eff * p_lk
    denominator = 1.0 - beta + beta * p_lk
    u_indiv = -c * numerator / denominator
    u_round = np.einsum('stlk,lk->stl', counts, u_indiv)
    weights = beta ** np.arange(T)
    U_per_seed = (u_round * weights[None, :, None]).sum(axis=1)
    return U_per_seed.mean(axis=0)  


def compute_or_load(model):
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        if int(d["n_seeds"]) == N_SEEDS and np.allclose(d["phi_grid"], PHI_GRID):
            return d["U"].item(), d["emp"].item()
    U = {}
    emp = {}
    for phi in PHI_GRID:
        for (lam9, lam16) in PROFILES:
            counts, e = simulate_profile(model, lam9, lam16, phi, N_SEEDS)
            emp[(round(phi, 3), lam9, lam16)] = e
            for beta in BETA_GRID:
                U[(round(phi, 3), lam9, lam16, round(beta, 3))] = \
                    prop_a1_utility(counts, model, lam9, lam16, beta)
    np.savez(CACHE, U=U, emp=emp, phi_grid=PHI_GRID, n_seeds=N_SEEDS)
    return U, emp


def main():
    model = build_full_model(LOCS_ALL)
    idx9  = LOCS_ALL.index(9)
    idx16 = LOCS_ALL.index(16)
    U, emp = compute_or_load(model)

    def inc_curves(beta):
        """At a given beta, return the 3 attack-incentive curves vs phi:
        (single Loc 9, Loc 9 given Loc 16 attacks, Loc 16 given Loc 9 attacks)."""
        b = round(beta, 3)
        s1, t9, t16 = [], [], []
        for phi in PHI_GRID:
            p = round(phi, 3)
            u_truth = U[(p, 0.0, 0.0, b)][idx9]
            best = max(U[(p, lam9, 0.0, b)][idx9] for lam9 in [0.5, 0.75, 1.0])
            s1.append(best - u_truth)
            t9.append(U[(p, 0.75, 0.75, b)][idx9]  - U[(p, 0.0, 0.75, b)][idx9])
            t16.append(U[(p, 0.75, 0.75, b)][idx16] - U[(p, 0.75, 0.0, b)][idx16])
        return np.array(s1), np.array(t9), np.array(t16)

    inc_single, inc_two_9, inc_two_16 = inc_curves(BETA)


    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(PHI_GRID, inc_single, marker="o", color="tab:blue", lw=2.6,
            markersize=10, markeredgecolor="white", markeredgewidth=1.1,
            label="1 attacker: Loc 9 alone")
    ax.plot(PHI_GRID, inc_two_9, marker="s", color="tab:orange", lw=2.6,
            markersize=9, markeredgecolor="white", markeredgewidth=1.0,
            linestyle="--",
            label="2 attackers: Loc 9 (given Loc 16 attacks)")
    ax.plot(PHI_GRID, inc_two_16, marker="^", color="tab:green", lw=2.6,
            markersize=9, markeredgecolor="white", markeredgewidth=1.0,
            linestyle="-.",
            label="2 attackers: Loc 16 (given Loc 9 attacks)")
    ax.axhline(0, color="gray", lw=1.0)
    ax.set_xlabel(r"Quota tightness  $\varphi$")
    ax.set_ylabel("Utility gain from attacking")
    ax.set_xticks(PHI_GRID)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(HERE, "step3_floor_defense.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


    emp_truth = np.array([emp[(round(phi, 3), 0.0, 0.0)] for phi in PHI_GRID])
    emp_loc9  = np.array([emp[(round(phi, 3), 1.0, 0.0)] for phi in PHI_GRID])
    emp_both  = np.array([emp[(round(phi, 3), 0.75, 0.75)] for phi in PHI_GRID])

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(PHI_GRID, emp_truth, marker="o", color="#2ca02c", lw=2.6,
            markersize=10, markeredgecolor="white", markeredgewidth=1.1,
            label="truthful (no attack)")
    ax.plot(PHI_GRID, emp_loc9, marker="s", color="tab:blue", lw=2.6,
            markersize=9, linestyle="--", markeredgecolor="white",
            markeredgewidth=1.0, label="Loc 9 attacks")
    ax.plot(PHI_GRID, emp_both, marker="^", color="#d62728", lw=2.6,
            markersize=9, linestyle="-.", markeredgecolor="white",
            markeredgewidth=1.0, label="Loc 9 + Loc 16 attack")
    ax.set_xlabel(r"Quota tightness  $\varphi$")
    ax.set_ylabel("Total employment rate")
    ax.set_xticks(PHI_GRID)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    out2 = os.path.join(HERE, "step3_floor_employment.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(figsize=(10, 5.4))
    cmap = plt.cm.viridis
    for bi, beta in enumerate(BETA_GRID):
        s1, _, _ = inc_curves(beta)
        ax.plot(PHI_GRID, s1, marker="o", lw=2.2, markersize=7,
                color=cmap(bi / (len(BETA_GRID) - 1)),
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"$\\beta = {beta:.1f}$")
    ax.axhline(0, color="gray", lw=1.0)
    ax.set_xlabel(r"Quota tightness  $\varphi$")
    ax.set_ylabel("Utility gain from attacking (Loc 9 alone)")
    ax.set_xticks(PHI_GRID)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.95, title="discount factor")
    plt.tight_layout()
    out3 = os.path.join(HERE, "step3_floor_beta_sensitivity.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
