# Heatmap analogue de plot_loc_asymmetry_real.py mais utilisant :
#   - utilite Proposition A.1 par individu (depend de p_{k, l} et lambda)
#   - somme cumulee escomptee avec discount beta sur les T rounds
# au lieu de l'ancienne formule (-2 + o) * c moyenne sur la seconde moitie.
#
# Ce script ne modifie pas l'ancienne figure (loc_asymmetry_real_14_4_strat_appendix.png).
# Il en cree une NOUVELLE (loc_asymmetry_propA1_14_4_beta<X>.png) pour la comparaison.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Font bump consistent with the other Section 5 figures.
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
})

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import build_2loc_model, find_pure_nash
from beta_sweep import (collect_assignments, aggregate_payoff,
                                   CACHE, N_SEEDS_SENS, ATTACK_GRID)


LOC_A, LOC_B = 14, 4
# Canonical reference value for all paper figures except the sensitivity plot
# (Fig 6) and its complementary panel: beta = 0.9. The same beta is applied
# (i) inside the Prop A.1 per-individual utility (Appendix A.1 future-cost
# integration) and (ii) outside as the discount across the T = 39
# simulation rounds (Section 3 expected discounted utility). beta = 1 was
# tried first but makes both layers undamped: per-individual cost loses its
# p-modulation (collapses to -c/p - c*lambda) and the discounted cumulative no
# longer decays, so U_l accumulates linearly over T rounds and no pure NE
# exists on the discrete grid. beta = 0.9 stays close to "fully patient"
# while preserving a clean pure NE at lambda* = (1, 1).
BETA = 0.9


def load_counts(model):
    """Load assignment counts collected by beta_sweep.collect_assignments
    (cached on disk to avoid re-running the LP)."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if int(d["n_seeds"]) == N_SEEDS_SENS:
            print(f"  Loaded assignment cache from {CACHE}")
            return d["counts"], int(d["T"])
    print("  Cache missing, computing assignment counts...")
    counts, T = collect_assignments(model, N_SEEDS_SENS)
    np.savez(CACHE, counts=counts, T=T, n_seeds=N_SEEDS_SENS)
    return counts, T


def plot_heatmap(loc_a, loc_b, beta):
    here = os.path.dirname(__file__)
    model = build_2loc_model(loc_a, loc_b)
    counts, T = load_counts(model)
    U0, U1 = aggregate_payoff(counts, model, beta)

    grid = ATTACK_GRID
    K = len(grid)
    tick = [f"{int(x * 100)}%" for x in grid]

    nash = find_pure_nash(U0, U1)
    cap, alpha = model["cap"], model["alpha"]

    vmin = min(U0.min(), U1.min())
    vmax = max(U0.max(), U1.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             gridspec_kw={"wspace": 0.30})

    panels = [
        (axes[0], U0, r"$U_{\ell_0}$",  True),
        (axes[1], U1, r"$U_{\ell_1}$",  False),
    ]
    last_im = None
    for ax, M, name, hide_xaxis in panels:
        im = ax.imshow(M, cmap="RdYlGn", origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_xticks(range(K))
        ax.set_yticks(range(K)); ax.set_yticklabels(tick)
        ax.set_ylabel(r"$\lambda_0$")
        if hide_xaxis:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xticklabels(tick)
            ax.set_xlabel(r"$\lambda_1$")
        ax.set_title(name)
        # cell values (rounded to 1 decimal since values are large at high beta)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=10, color="black")
        # mark pure NE cell(s) with a thick black border
        for (i, j) in nash:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="black", lw=3.0, zorder=5))

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.02)

    out = os.path.join(here,
                       f"loc_asymmetry_propA1_{loc_a}_{loc_b}_beta{int(beta*100):02d}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")

    # ready-to-paste LaTeX caption -------------------------------------
    if nash:
        i_ne, j_ne = nash[0]
        ne_pair = f"({grid[i_ne]:.2f}, {grid[j_ne]:.2f})"
        u_pair  = f"({U0[i_ne, j_ne]:+.1f},\\, {U1[i_ne, j_ne]:+.1f})"
    else:
        ne_pair = "n/a"
        u_pair  = "no pure NE on grid"
    print("\n  Suggested LaTeX caption (paste under the figure):\n")
    print(
        "  \\caption{Strategy-aware utility heatmaps "
        f"$U_{{\\ell_0}}$ (left) and $U_{{\\ell_1}}$ (right) under the "
        f"generalised utility of Proposition~A.1 with discount factor "
        f"$\\beta = {beta}$. Each cell is the expected cumulative discounted "
        f"utility $U_\\ell = \\mathbb{{E}}[\\sum_{{t=0}}^{{T-1}} \\beta^t "
        f"\\sum_{{i \\in I^t_\\ell}} u_{{i,\\ell,\\lambda}}]$ over $T = {T}$ "
        f"quarters, averaged across $N = {N_SEEDS_SENS}$ Monte Carlo seeds. "
        f"The thick black border marks the pure Nash equilibrium at "
        f"$\\lambda^\\star = {ne_pair}$ with payoff $U^\\star = {u_pair}$. "
        f"Setting: $\\ell_0 = \\text{{Loc {loc_a}}}$ (cap $= {int(cap[0])}$, "
        f"$\\alpha_0 = {alpha[0]:.2f}$), "
        f"$\\ell_1 = \\text{{Loc {loc_b}}}$ (cap $= {int(cap[1])}$, "
        f"$\\alpha_1 = {alpha[1]:.2f}$).}}"
    )


if __name__ == "__main__":
    plot_heatmap(LOC_A, LOC_B, BETA)
