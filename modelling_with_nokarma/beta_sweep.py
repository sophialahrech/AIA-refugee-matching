# Beta sweep for the 2-location attack game (locs 14 and 4).
# For each beta: build the payoff matrices, find the pure Nash
# equilibrium, and plot the NE attack levels and the attack gain vs beta.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import (build_2loc_model, location_attacks, solve_lp,
                        round_assignment, ATTACK_GRID, find_pure_nash,
                        N_TYPES, TYPE_PROBS, C_APPENDIX)


plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11.5,
})



LOC_A, LOC_B = 14, 4
N_SEEDS_SENS = 100
BETA_GRID = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])

CACHE = os.path.join(os.path.dirname(__file__),
                     f"beta_sweep_propA1_{LOC_A}_{LOC_B}.npz")


def collect_assignments(model, n_seeds):
    """tells us how many refugees of each type were assigned to each location, 
    per round, per seed, for each attack profile on the grid."""
    arrivals = model["quarterly_arrivals"]
    T = len(arrivals)
    n_loc = 2
    cost_mask = model["cost_mask"]
    alpha = model["alpha"]
    theta = model["theta"]

    K = len(ATTACK_GRID)
    counts = np.zeros((K, K, n_seeds, T, n_loc, N_TYPES)) # (level of attack l0, level of attack l1, seed, quarter, loc, type) 

    for i, lam0 in enumerate(ATTACK_GRID):
        for j, lam1 in enumerate(ATTACK_GRID):
            attack = np.array([lam0, lam1])
            for s in range(n_seeds):
                np.random.seed(s)
                for t in range(T): # loop over quarters
                    n_t = int(arrivals[t])
                    if n_t == 0:
                        continue
                    types = np.random.choice(N_TYPES, size=n_t, p=TYPE_PROBS)
                    V_true = theta[types]
                    lambdas = location_attacks(types, attack, cost_mask)
                    # strat-aware planner: knows lambda, uses (1-lambda) V_true
                    V_strat = (1 - lambdas) * V_true
                    a_S = round_assignment(solve_lp(V_strat, alpha)).argmax(axis=1)
                    for ii in range(n_t):
                        counts[i, j, s, t, a_S[ii], types[ii]] += 1
    return counts, T


def load_or_collect(model):
    
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if int(d["n_seeds"]) == N_SEEDS_SENS:
            return d["counts"], int(d["T"])
    counts, T = collect_assignments(model, N_SEEDS_SENS)
    np.savez(CACHE, counts=counts, T=T, n_seeds=N_SEEDS_SENS)
    return counts, T


def aggregate_payoff(counts, model, beta, c=C_APPENDIX):
    """turns the assignment counts into each location's payoff for one beta:
    give every refugee its Prop A.1 utility."""
    K_grid, _, n_seeds, T, n_loc, n_types = counts.shape
    theta     = model["theta"]            # (N_TYPES, n_loc)
    cost_mask = model["cost_mask"]        # (N_TYPES, n_loc) 

   
    lambda_eff = np.zeros((K_grid, K_grid, n_loc, n_types))
    for i in range(K_grid):
        for j in range(K_grid):
            for l in range(n_loc):
                lam_l = ATTACK_GRID[i] if l == 0 else ATTACK_GRID[j]
                for k in range(n_types):
                    lambda_eff[i, j, l, k] = lam_l * (1.0 if cost_mask[k, l] else 0.0)

    #  probability p_{k, l}
    p_lk = theta.T                                 
    p_broadcast = p_lk[None, None, :, :]        

    #  expected discounted utility
    numerator   = 1.0 + beta * lambda_eff * p_broadcast
    denominator = 1.0 - beta + beta * p_broadcast
    u_indiv = -c * numerator / denominator


    #  per-round utility = sum over types of counts * u_indiv
    u_round = np.einsum('ijstlk,ijlk->ijstl', counts, u_indiv)

    # discounted sum over quarters
    weights = beta ** np.arange(T)  
    U_per_seed = (u_round * weights[None, None, None, :, None]).sum(axis=3)


    U = U_per_seed.mean(axis=2) 
    return U[..., 0], U[..., 1]


def main():
    model = build_2loc_model(LOC_A, LOC_B)
    counts, T = load_or_collect(model)


    betas, ne_lam0, ne_lam1, gain_L0, gain_L1 = [], [], [], [], []
    for beta in BETA_GRID:
        U0, U1 = aggregate_payoff(counts, model, beta)
        nash = find_pure_nash(U0, U1)
        if nash:
            i_ne, j_ne = nash[0]
            ne_lam0.append(ATTACK_GRID[i_ne])
            ne_lam1.append(ATTACK_GRID[j_ne])
        else:
            ne_lam0.append(np.nan)
            ne_lam1.append(np.nan)
        betas.append(beta)
        gain_L0.append(U0[-1, -1] - U0[0, 0])
        gain_L1.append(U1[-1, -1] - U1[0, 0])

    betas   = np.array(betas)
    ne_lam0 = np.array(ne_lam0)
    ne_lam1 = np.array(ne_lam1)
    gain_L0 = np.array(gain_L0)
    gain_L1 = np.array(gain_L1)

    OFFSET = 0.02   

    fig, ax = plt.subplots(figsize=(11, 4.2))

    ax.plot(betas, ne_lam0, marker="o", color="tab:blue", lw=2.2,
            markersize=8, markeredgecolor="white", markeredgewidth=1.0,
            label=r"$\lambda^\star_0$  (Loc $\ell_0$)")
    ax.plot(betas, ne_lam1 + OFFSET, marker="s", color="tab:orange",
            lw=2.2, markersize=8, markeredgecolor="white",
            markeredgewidth=1.0, linestyle="--",
            label=r"$\lambda^\star_1$  (Loc $\ell_1$)")


    ax.axvline(0.5, color="gray", linestyle=":", lw=1.1, alpha=0.7)

    ax.set_xlabel(r"Discount factor $\beta$")
    ax.set_ylabel(r"NE attack level $\lambda^\star_\ell$")
    ax.set_ylim(-0.12, 1.18)
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "beta_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

   
    # Second figure: utility gain from full attack vs truthful
    fig2, ax2 = plt.subplots(figsize=(11, 4.2))

    ax2.plot(betas, gain_L0, marker="o", color="tab:blue", lw=2.4,
             markersize=8, markeredgecolor="white", markeredgewidth=1.0,
             label=r"$\Delta U_{\ell_0}$ (Loc $\ell_0$)")
    ax2.plot(betas, gain_L1, marker="s", color="tab:orange", lw=2.4,
             markersize=8, markeredgecolor="white", markeredgewidth=1.0,
             linestyle="--",
             label=r"$\Delta U_{\ell_1}$ (Loc $\ell_1$)")

    ax2.axhline(0.0, color="gray", linestyle="-", lw=0.9, alpha=0.6)

    ax2.set_xlabel(r"Discount factor $\beta$")
    ax2.set_ylabel(r"Attack gain  $\Delta U_\ell = U_\ell(1,1) - U_\ell(0,0)$")
    ax2.set_xlim(0.0, 1.05)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__),
                        "beta_sweep_utility.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
