# MPC trajectory figure (Fig 5 analogue) computed under the generalised
# utility of Proposition A.1 combined with the full discounted cumulative
# discount across all T simulation rounds.
#
# Two stacked panels:
#   (a) IBR  (H = 1) -- async iterative best response on the payoff matrix
#                        computed with Prop A.1 + discounted cumulative
#   (b) Bilevel MPC-BR (H >= 2) -- direct jump to the pure Nash equilibrium
#                                   of the same payoff matrix
#
# Differences vs plot_mpc_br_real.py:
#   - payoff matrix uses Prop A.1 per-individual utility (depends on the
#     baseline probability p_{k,l} and the attack level lambda)
#   - aggregation is the discounted cumulative sum over all T rounds, not
#     the per-round average over the second half
#   - the same beta drives both (per-individual + discounted)
#
# This script does NOT touch plot_mpc_br_real.py or its output figure.
# Output: mpc_br_propA1_beta<XX>.png

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
})

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import build_2loc_model, find_pure_nash
from beta_sweep import (collect_assignments, aggregate_payoff,
                                   CACHE, N_SEEDS_SENS, ATTACK_GRID)
# Reuse the same async IBR routine that produced Fig 5(a) in the paper
from ibr_common import best_response_async


LOC_A, LOC_B = 14, 4
# Canonical reference value for all paper figures except the sensitivity plot
# (Fig 6) and its complementary panel: beta = 0.9. The same beta is applied
# (i) inside the Prop A.1 per-individual utility (Appendix A.1 future-cost
# integration) and (ii) outside as the discount across the T = 39
# simulation rounds. beta = 1 was tried first but makes both layers undamped
# and removes the pure NE from the grid; beta = 0.9 stays close to "fully
# patient" while preserving lambda* = (1, 1) as a clean pure NE.
BETA = 0.9


def load_counts(model):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if int(d["n_seeds"]) == N_SEEDS_SENS:
            print(f"  Loaded assignment cache from {CACHE}")
            return d["counts"], int(d["T"])
    print("  Cache missing, computing assignment counts...")
    counts, T = collect_assignments(model, N_SEEDS_SENS)
    np.savez(CACHE, counts=counts, T=T, n_seeds=N_SEEDS_SENS)
    return counts, T


def best_response_mpc_bilevel(U0, U1, start=(0, 0)):
    """Bilevel MPC-BR : in a static 2-player game with common knowledge,
    H >= 2 collapses to the pure Nash equilibrium of (U0, U1). We commit
    to it in a single synchronous round."""
    nash = find_pure_nash(U0, U1)
    if not nash:
        return [start], "no_pure_NE"
    return [start, nash[0]], "fixed"


def _step_xs_ys(traj, grid, dim):
    xs = np.arange(len(traj))
    ys = np.array([grid[c[dim]] for c in traj])
    return xs, ys


def plot_mpc(loc_a, loc_b, beta):
    here = os.path.dirname(__file__)
    model = build_2loc_model(loc_a, loc_b)
    counts, T = load_counts(model)
    U0, U1 = aggregate_payoff(counts, model, beta)

    grid = ATTACK_GRID
    nash = find_pure_nash(U0, U1)
    print(f"  beta = {beta}")
    print(f"  pure NE on Prop A.1 payoff: {nash}  "
          f"lam* = {[(grid[i], grid[j]) for (i, j) in nash]}")

    # IBR (H = 1)
    traj_ibr, _, status_ibr = best_response_async(U0, U1)
    print(f"  IBR : status={status_ibr}, length={len(traj_ibr)}, "
          f"cells={[(grid[i], grid[j]) for (i, j) in traj_ibr]}")

    # MPC-BR (H >= 2)
    traj_mpc, status_mpc = best_response_mpc_bilevel(U0, U1)
    print(f"  MPC-BR (bilevel): status={status_mpc}, length={len(traj_mpc)}, "
          f"cells={[(grid[i], grid[j]) for (i, j) in traj_mpc]}")

    # plot ---------------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 7.5),
                                     gridspec_kw={"hspace": 0.32},
                                     sharex=True)

    # (a) IBR
    xs_a, ys0_a = _step_xs_ys(traj_ibr, grid, 0)
    _,    ys1_a = _step_xs_ys(traj_ibr, grid, 1)
    ax_a.step(xs_a, ys0_a, where="post", marker="o", color="tab:blue",
              lw=2.6, markersize=10, markeredgecolor="white",
              markeredgewidth=1, label=f"$\\lambda_0$  (Loc {LOC_A})")
    ax_a.step(xs_a, ys1_a, where="post", marker="s", color="tab:orange",
              lw=2.6, markersize=10, markeredgecolor="white",
              markeredgewidth=1, linestyle="--",
              label=f"$\\lambda_1$  (Loc {LOC_B})")
    ax_a.set_ylim(-0.10, 1.20)
    ax_a.set_ylabel(r"Attack level $\lambda_\ell$")
    ax_a.set_xticks(xs_a)
    ax_a.grid(alpha=0.3)
    ax_a.set_title(r"(a)  IBR  ($H=1$)", loc="left")
    ax_a.legend(loc="center right")

    # (b) Bilevel MPC-BR
    xs_b, ys0_b = _step_xs_ys(traj_mpc, grid, 0)
    _,    ys1_b = _step_xs_ys(traj_mpc, grid, 1)
    ax_b.step(xs_b, ys0_b, where="post", marker="o", color="tab:blue",
              lw=3.0, markersize=13, markeredgecolor="white",
              markeredgewidth=1.3)
    ax_b.step(xs_b, ys1_b, where="post", marker="s", color="tab:orange",
              lw=3.0, markersize=13, markeredgecolor="white",
              markeredgewidth=1.3, linestyle="--")
    ax_b.set_ylim(-0.10, 1.20)
    ax_b.set_xlim(ax_a.get_xlim())
    ax_b.set_xticks(np.arange(int(ax_a.get_xlim()[1]) + 1))
    ax_b.set_xlabel("Best-response round")
    ax_b.set_ylabel(r"Attack level $\lambda_\ell$")
    ax_b.grid(alpha=0.3)
    ax_b.set_title(r"(b)  Bilevel MPC-BR  ($H \geq 2$)", loc="left")

    # discreet NE reference lines on both panels
    if nash:
        i_ne, j_ne = nash[0]
        for ax in (ax_a, ax_b):
            ax.axhline(grid[i_ne], color="tab:blue",   ls=":", lw=1.0, alpha=0.5)
            ax.axhline(grid[j_ne], color="tab:orange", ls=":", lw=1.0, alpha=0.5)

    plt.tight_layout()
    out = os.path.join(here,
                       f"mpc_br_propA1_beta{int(beta*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")



if __name__ == "__main__":
    plot_mpc(LOC_A, LOC_B, BETA)
