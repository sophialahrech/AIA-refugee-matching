# IBR trajectory figure: employment (total and per type) along the iterative
# best-response path from truthful (0,0) to the Nash equilibrium, on the
# Prop A.1 payoff matrix (locs 14 and 4).
#
# Output: ibr_trajectory_propA1_beta<XX>.png

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
    "legend.fontsize":  11.5,
})

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import build_2loc_model, find_pure_nash, N_TYPES
from beta_sweep import aggregate_payoff, ATTACK_GRID, load_or_collect
# reuse the same IBR routine and per-cell simulator already used in Fig 4
from ibr_common import (best_response_async, simulate_cell, TYPE_NAMES,
                        N_SEEDS_TRAJ)


LOC_A, LOC_B = 14, 4
BETA = 0.9   # used both inside the Prop A.1 utility and as the discount over quarters


def main():
    print("=" * 78)
    print(f"IBR trajectory (Prop A.1, discounted cumulative)  pair "
          f"({LOC_A}, {LOC_B})  beta={BETA}")
    print("=" * 78)

    model = build_2loc_model(LOC_A, LOC_B)
    counts, T = load_or_collect(model)
    U0, U1 = aggregate_payoff(counts, model, BETA)
    grid = ATTACK_GRID

    nash = find_pure_nash(U0, U1)
    print(f"  pure NE on Prop A.1 payoff: {nash}  "
          f"lam* = {[(grid[i], grid[j]) for (i, j) in nash]}")

    traj, cycle_start, status = best_response_async(U0, U1)
    print(f"  IBR ended: status={status}, length={len(traj)}, "
          f"trajectory = {[(grid[i], grid[j]) for (i, j) in traj]}")

    # Per-cell simulation for per-type employment (uses run_one_seed which
    # samples arrivals, runs the LP and Bernoulli outcomes; employment is
    # independent of the utility formula).
    print(f"\n  Re-simulating {len(traj)} cells with {N_SEEDS_TRAJ} seeds...")
    cells = []
    for k_step, (i, j) in enumerate(traj):
        lam = np.array([grid[i], grid[j]])
        emp, per_type, emp_std, pt_std = simulate_cell(model, lam,
                                                       N_SEEDS_TRAJ)
        cells.append(dict(step=k_step, lam=lam.copy(),
                          emp=emp, emp_std=emp_std,
                          per_type=per_type.copy(), pt_std=pt_std.copy()))
        print(f"    step {k_step}: lam=({lam[0]:.2f}, {lam[1]:.2f})  "
              f"emp={emp:.3f} +/- {emp_std:.3f}  "
              f"per_type={np.round(per_type, 2).tolist()}")

    # After-attack state ----------------------------------------------------
    pt0 = cells[0]["per_type"]
    if status == "cycle":
        cycle_cells = cells[cycle_start:-1]
        end_label = f"cycle (avg over {len(cycle_cells)} cells)"
        pt_end = np.nanmean(np.stack([c["per_type"] for c in cycle_cells]),
                            axis=0)
        e_end  = float(np.mean([c["emp"] for c in cycle_cells]))
    else:
        end_label = "fixed point"
        pt_end = cells[-1]["per_type"]
        e_end  = cells[-1]["emp"]
    e0 = cells[0]["emp"]
    drop = pt0 - pt_end
    drop[np.isnan(drop)] = 0
    worst_type = int(np.nanargmax(drop))
    pct_drop = 100 * (e0 - e_end) / max(e0, 1e-9)

    print(f"\n  After-attack state : {end_label}")
    print(f"  Type with largest drop: type {worst_type} "
          f"({TYPE_NAMES[worst_type]})  drop={drop[worst_type]:+.3f}")
    print(f"  Total employment: truthful={e0:.3f} -> after-attack={e_end:.3f}  "
          f"absolute drop={e0 - e_end:+.4f}  "
          f"relative drop={pct_drop:+.1f}%")

    # Plot ------------------------------------------------------------------
    steps     = np.arange(len(traj))
    lam0_path = np.array([c["lam"][0] for c in cells])
    lam1_path = np.array([c["lam"][1] for c in cells])
    emp_path  = np.array([c["emp"] for c in cells])
    emp_std   = np.array([c["emp_std"] for c in cells])
    pt_paths  = np.array([c["per_type"] for c in cells])
    pt_stds   = np.array([c["pt_std"]   for c in cells])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.10},
    )

    DROP_THRESH = 0.04
    affected = np.abs(drop) >= DROP_THRESH
    STABLE_REPRESENTATIVE = 7
    STABLE_INTERESTING   = 5
    show_mask = affected.copy()
    show_mask[STABLE_REPRESENTATIVE] = True
    show_mask[STABLE_INTERESTING]    = True

    cmap = plt.cm.tab10
    for affected_now in (False, True):
        for k in range(N_TYPES):
            if not show_mask[k]:
                continue
            if affected[k] != affected_now:
                continue
            line = pt_paths[:, k]
            err  = pt_stds[:, k]
            if np.all(np.isnan(line)):
                continue
            col = cmap(k)
            is_worst = (k == worst_type)
            if is_worst:
                style = dict(marker="o", markersize=10, lw=3.0, alpha=1.0,
                             zorder=7, capsize=4, elinewidth=1.5,
                             markeredgecolor="white", markeredgewidth=1.2)
            elif affected_now:
                style = dict(marker="o", markersize=7, lw=2.0, alpha=0.90,
                             zorder=5, capsize=3, elinewidth=1.0,
                             markeredgecolor="white", markeredgewidth=0.8)
            else:
                style = dict(marker="s", markersize=6, lw=1.8, alpha=0.85,
                             zorder=3, capsize=2.5, elinewidth=0.9)
            label = f"type {k}: {TYPE_NAMES[k]}  ($\\Delta={drop[k]:+.2f}$)"
            if is_worst:
                label += "  WORST"
            ax.errorbar(steps, line, yerr=err, color=col, label=label, **style)

    xtick_labels = [
        f"$\\lambda$=({l0:.2f}, {l1:.2f})\nstep {k}"
        for k, (l0, l1) in enumerate(zip(lam0_path, lam1_path))
    ]
    ax.set_ylabel("Employment rate by type\n(mean $\\pm$ std across seeds)")
    ax.set_ylim(-0.03, 0.72)
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    def _key(lbl):
        try:
            k = int(lbl.split(":")[0].replace("type", "").strip())
            return (1, -abs(drop[k]))
        except Exception:
            return (3, 0)
    order = sorted(range(len(labels)), key=lambda i: _key(labels[i]))
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    n_legend = len(labels)
    ncol_legend = 3 if n_legend <= 6 else 4
    ax.legend(handles, labels,
              loc="lower center", bbox_to_anchor=(0.5, 1.04),
              ncol=ncol_legend, framealpha=1.0, borderaxespad=0.3,
              handlelength=2.0, columnspacing=1.4, handletextpad=0.6)

    ax2.axhline(e0, color="#1f77b4", ls="--", lw=2.5, alpha=0.85,
                zorder=2,
                label=r"truthful baseline $E_0$")
    ax2.errorbar(steps, emp_path, yerr=emp_std,
                 marker="D", color="black", lw=4.5, markersize=15,
                 capsize=7, elinewidth=2.4, alpha=1.0,
                 markeredgecolor="white", markeredgewidth=2.2,
                 zorder=5,
                 label=r"TOTAL employment $E$")

    ax2.set_xticks(steps)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xlabel(r"Iterative best-response trajectory  "
                   r"$(\lambda_0,\, \lambda_1)$  at each step")
    ax2.set_ylabel("Total employment rate\n(mean $\\pm$ std across seeds)")
    ax2.set_ylim(0.0, 0.5)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", framealpha=1.0)

    plt.tight_layout(rect=[0, 0, 1.0, 0.92])
    out = os.path.join(os.path.dirname(__file__),
                       f"ibr_trajectory_propA1_beta{int(BETA*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out}")

    # caption ---------------------------------------------------------------
    affected_k = [k for k in range(N_TYPES) if model["cost_mask"][k].all()]
    affected_str = ",".join(str(k) for k in affected_k)
    print("\n  Suggested LaTeX caption (paste under the figure):\n")
    print(
        "  \\caption{Per-type (top) and aggregate (bottom) employment along "
        "the asynchronous iterative best-response trajectory from truthful "
        "$\\lambda = (0,0)$ to the pure Nash equilibrium "
        f"$\\lambda^\\star = (1,1)$ on the Prop~A.1 payoff matrix "
        f"($\\beta = {BETA}$), reached in {len(traj)-1} steps. Aggregate "
        f"employment falls from $E_0 = {e0:.3f}$ to "
        f"$E_{{\\rm NE}} = {e_end:.3f}$, a relative decrease of "
        f"${pct_drop:.1f}\\%$. The loss is concentrated on the types in "
        f"the cost-mask at both locations ($k={affected_str}$); the "
        f"worst-affected is $k={worst_type}$ "
        f"(\\textit{{{TYPE_NAMES[worst_type]}}}), losing "
        f"{drop[worst_type]*100:.0f} percentage points. The remaining types "
        "are unaffected and a representative subset is shown for clarity. "
        f"All curves averaged across $N={N_SEEDS_TRAJ}$ Monte~Carlo seeds.}}"
    )


if __name__ == "__main__":
    main()
