# game-theoretic analysis des attaques - MPC version
# on suit la methodo de Ionescu et al. 2023 (NeurIPS), Sec 5.3 / Fig 5a
#
# The per-cell utility U_l(lambda_0, lambda_1) is the average over the second
# half of the rounds (MPC with horizon H = N_ROUNDS/2). 

import os
import numpy as np
import matplotlib.pyplot as plt

import simu2


N_SEEDS = 20
N_ROUNDS = 40
ATTACK_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
HALF_AVG = True          
MPC_HORIZON = N_ROUNDS // 2  
CACHE_PATH = "payoff_matrix_cache_v3.npz"   # v3 = random tie-breaking in LP + round_assignment


def patch_globals():
    simu2.n_seeds = N_SEEDS
    simu2.n_rounds = N_ROUNDS


def run_scenario(lambda_0, lambda_1, n_seeds=None):
   # runs simu2 for the pair (lambda_0, lambda_1) and returns
# the steady-state utilities and employment rates for the three oracles:
#   - learned: trained predictor (noisy; this is what we actually observe)
#   - true oracle: knows V_true, but is still affected by the attacks
#   - strat oracle: knows V_true and lambda ->upper bound
    if n_seeds is None:
        n_seeds = N_SEEDS
    attack_vec = np.array([lambda_0, lambda_1])
    util_learned = np.zeros((n_seeds, simu2.n_locations))
    util_oracle = np.zeros((n_seeds, simu2.n_locations))
    util_strat = np.zeros((n_seeds, simu2.n_locations))
    emp_learned = np.zeros(n_seeds)
    emp_oracle = np.zeros(n_seeds)
    emp_strat = np.zeros(n_seeds)

    for s in range(n_seeds):
        (el, eo, es, _, ul, uo, us) = simu2.run_one_seed(
            seed=s, attack_levels_vec=attack_vec
        )
        if HALF_AVG:
            half = simu2.n_rounds // 2
            util_learned[s] = ul[half:].mean(axis=0)
            util_oracle[s] = uo[half:].mean(axis=0)
            util_strat[s] = us[half:].mean(axis=0)
            emp_learned[s] = el[half:].mean()
            emp_oracle[s] = eo[half:].mean()
            emp_strat[s] = es[half:].mean()
        else:
            util_learned[s] = ul.mean(axis=0)
            util_oracle[s] = uo.mean(axis=0)
            util_strat[s] = us.mean(axis=0)
            emp_learned[s] = el.mean()
            emp_oracle[s] = eo.mean()
            emp_strat[s] = es.mean()
    return (util_learned, util_oracle, util_strat,
            emp_learned, emp_oracle, emp_strat)


def compute_payoff_matrix(use_cache=True):

    K = len(ATTACK_GRID)
    required = {"grid", "n_seeds", "n_rounds",
                "U0_learned", "U1_learned", "U0_oracle", "U1_oracle",
                "U0_strat",   "U1_strat",
                "E_learned",  "E_oracle",  "E_strat"}

    if use_cache and os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        if (required.issubset(set(d.files))
                and d["grid"].shape == ATTACK_GRID.shape
                and np.allclose(d["grid"], ATTACK_GRID)
                and int(d["n_seeds"]) == N_SEEDS
                and int(d["n_rounds"]) == N_ROUNDS):
            print(f"  Loaded cached payoff matrix from {CACHE_PATH}")
            return (d["U0_learned"], d["U1_learned"],
                    d["U0_oracle"],  d["U1_oracle"],
                    d["U0_strat"],   d["U1_strat"],
                    d["E_learned"],  d["E_oracle"],  d["E_strat"])
        else:
            print(f"  Cache at {CACHE_PATH} is stale, recomputing.")

    U0_learned = np.zeros((K, K))
    U1_learned = np.zeros((K, K))
    U0_oracle  = np.zeros((K, K))
    U1_oracle  = np.zeros((K, K))
    U0_strat   = np.zeros((K, K))
    U1_strat   = np.zeros((K, K))
    E_learned  = np.zeros((K, K))
    E_oracle   = np.zeros((K, K))
    E_strat    = np.zeros((K, K))

    total = K * K
    k = 0
    for i, lam0 in enumerate(ATTACK_GRID):
        for j, lam1 in enumerate(ATTACK_GRID):
            k += 1
            print(f"  [{k:2d}/{total}] lambda = ({lam0:.2f}, {lam1:.2f}) ...",
                  end="", flush=True)
            ul, uo, us, el, eo, es = run_scenario(lam0, lam1)
            U0_learned[i, j] = ul[:, 0].mean()
            U1_learned[i, j] = ul[:, 1].mean()
            U0_oracle[i, j]  = uo[:, 0].mean()
            U1_oracle[i, j]  = uo[:, 1].mean()
            U0_strat[i, j]   = us[:, 0].mean()
            U1_strat[i, j]   = us[:, 1].mean()
            E_learned[i, j]  = el.mean()
            E_oracle[i, j]   = eo.mean()
            E_strat[i, j]    = es.mean()
            print(f"  E_learned={E_learned[i,j]:.3f}"
                  f"  E_oracle={E_oracle[i,j]:.3f}"
                  f"  E_strat={E_strat[i,j]:.3f}")

    np.savez(CACHE_PATH,
             grid=ATTACK_GRID, n_seeds=N_SEEDS, n_rounds=N_ROUNDS,
             U0_learned=U0_learned, U1_learned=U1_learned,
             U0_oracle=U0_oracle,   U1_oracle=U1_oracle,
             U0_strat=U0_strat,     U1_strat=U1_strat,
             E_learned=E_learned,   E_oracle=E_oracle,   E_strat=E_strat)
    print(f"  Cached payoff matrix to {CACHE_PATH}")
    return (U0_learned, U1_learned, U0_oracle, U1_oracle,
            U0_strat,   U1_strat,
            E_learned,  E_oracle,  E_strat)


def best_response_async(U0, U1, max_iters=30, noise_sigma=0.0, noise_seed=0):
    # Greedy async best response: at each turn, the moving
    # player sees the full payoff row/column and jumps directly to its
    # argmax cell. 
    if noise_sigma > 0:
        rng = np.random.default_rng(noise_seed)
        U0 = U0 + rng.normal(0.0, noise_sigma, size=U0.shape)
        U1 = U1 + rng.normal(0.0, noise_sigma, size=U1.shape)
    i, j = 0, 0
    traj = [(i, j)]
    for t in range(max_iters):
        i_new = int(np.argmax(U0[:, j]))
        changed_a = (i_new != i)
        i = i_new
        if (i, j) in traj and changed_a:
            traj.append((i, j))
            return traj
        if changed_a:
            traj.append((i, j))

        j_new = int(np.argmax(U1[i, :]))
        changed_b = (j_new != j)
        j = j_new
        if (i, j) in traj and changed_b:
            traj.append((i, j))
            return traj
        if changed_b:
            traj.append((i, j))

        if not changed_a and not changed_b:
            break
    return traj



def is_nash(U0, U1, i, j):
    return (U0[i, j] == U0[:, j].max()) and (U1[i, j] == U1[i, :].max())


def find_all_pure_nash(U0, U1):
    K = U0.shape[0]
    eq = []
    for i in range(K):
        for j in range(K):
            if is_nash(U0, U1, i, j):
                eq.append((i, j))
    return eq


def print_payoff_matrix(U0, U1, title):
    K = len(ATTACK_GRID)
    print(f"\n  {title}")
    print("  " + "-" * 80)
    header = "  lambda_0\\lambda_1 |"
    for j in range(K):
        header += f"   {int(ATTACK_GRID[j]*100):3d}%   "
    print(header)
    print("  " + "-" * 80)
    for i in range(K):
        row = f"        {int(ATTACK_GRID[i]*100):3d}%       |"
        for j in range(K):
            row += f" ({U0[i,j]:+.2f},{U1[i,j]:+.2f})"
        print(row)
    print("  " + "-" * 80)


def binary_scenario_report(U0, U1, label):
    # les 4 scenarios  (none/L0/L1/both)
    K = len(ATTACK_GRID)
    i_off, j_off = 0, 0
    i_on, j_on = K - 1, K - 1
    print(f"\n  === Binary scenarios ({label}) ===")
    print(f"    neither   (0,0):   U = ({U0[i_off, j_off]:+.3f}, {U1[i_off, j_off]:+.3f})")
    print(f"    only L0   (1,0):   U = ({U0[i_on, j_off]:+.3f}, {U1[i_on, j_off]:+.3f})")
    print(f"    only L1   (0,1):   U = ({U0[i_off, j_on]:+.3f}, {U1[i_off, j_on]:+.3f})")
    print(f"    both      (1,1):   U = ({U0[i_on, j_on]:+.3f}, {U1[i_on, j_on]:+.3f})")
    print(f"    shift from (0,0):")
    print(f"      only L0 attacks:  dU_L0 = {U0[i_on, j_off]-U0[i_off, j_off]:+.3f},  "
          f"dU_L1 = {U1[i_on, j_off]-U1[i_off, j_off]:+.3f}")
    print(f"      only L1 attacks:  dU_L0 = {U0[i_off, j_on]-U0[i_off, j_off]:+.3f},  "
          f"dU_L1 = {U1[i_off, j_on]-U1[i_off, j_off]:+.3f}")
    print(f"      both attack:      dU_L0 = {U0[i_on, j_on]-U0[i_off, j_off]:+.3f},  "
          f"dU_L1 = {U1[i_on, j_on]-U1[i_off, j_off]:+.3f}")


def report_equilibria(U0, U1, label):
    pure = find_all_pure_nash(U0, U1)
    print(f"\n  === Pure Nash Equilibria ({label}) ===")
    if not pure:
        print("    (no pure NE found in the 5x5 grid)")
    for (i, j) in pure:
        print(f"    sigma = ({int(ATTACK_GRID[i]*100):3d}%, "
              f"{int(ATTACK_GRID[j]*100):3d}%)   "
              f"U = ({U0[i,j]:+.3f}, {U1[i,j]:+.3f})")
    print(f"    Truthful (0%,0%):  U = ({U0[0,0]:+.3f}, {U1[0,0]:+.3f})   "
          "[Pareto ref]")





def plot_paper_style(U0, U1, traj, label, out_path):
    # (a) BR trajectory in the space (U_0, U_1)
    # (b) incentive to deviate from truthful: DeltaU_l(lambda) = U_l(lambda,0) - U_l(0,0)
    from matplotlib.lines import Line2D

    seen = {}
    unique_traj = []
    for k, p in enumerate(traj):
        if p not in seen:
            seen[p] = k
            unique_traj.append(p)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Strategic-attack dynamics  —  {label} case",
                 fontsize=13, fontweight="bold")

    # left panel: BR trajectory
    ax_l.scatter(U0.flatten(), U1.flatten(), c="lightgray", s=25,
                 alpha=0.4, zorder=1)

    i0, j0 = unique_traj[0]
    ax_l.scatter([U0[i0, j0]], [U1[i0, j0]], color="black", s=130,
                 zorder=6, edgecolor="white", linewidth=1.5)

    for k in range(len(unique_traj) - 1):
        iA, jA = unique_traj[k]
        iB, jB = unique_traj[k + 1]
        if iA != iB:
            color = "tab:blue"      # L0 mooved
        else:
            color = "tab:green"     # L1 mooved
        ax_l.annotate("",
                      xy=(U0[iB, jB], U1[iB, jB]),
                      xytext=(U0[iA, jA], U1[iA, jA]),
                      arrowprops=dict(arrowstyle="->", color=color,
                                      lw=2.5, alpha=0.85))
        ax_l.scatter([U0[iB, jB]], [U1[iB, jB]], color=color, s=110,
                     zorder=5, edgecolor="white", linewidth=1.2)

    n_traj = len(unique_traj)
    for k, (ii, jj) in enumerate(unique_traj):
        col = "black" if k == 0 else (
            "tab:blue" if unique_traj[k - 1][0] != ii else "tab:green"
        )
        if k == 0 or k == n_traj - 1:

            lbl = f"({int(ATTACK_GRID[ii]*100)},{int(ATTACK_GRID[jj]*100)})"
            dx = 10 if k == 0 else -12
            dy = 10 if k == 0 else -14
            ha = "left" if k == 0 else "right"
            ax_l.annotate(lbl, (U0[ii, jj], U1[ii, jj]),
                          textcoords="offset points", xytext=(dx, dy),
                          ha=ha, fontsize=10, color=col, fontweight="bold")
        else:

            ax_l.annotate(f"{k}", (U0[ii, jj], U1[ii, jj]),
                          textcoords="offset points", xytext=(6, 6),
                          fontsize=8, color=col, fontweight="bold", alpha=0.9)

    handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="",
               markersize=10, label="Initial scenario (0%,0%)"),
        Line2D([0], [0], color="tab:blue",  lw=2.5, label="L0 responds"),
        Line2D([0], [0], color="tab:green", lw=2.5, label="L1 responds"),
    ]
    leg_colors = ax_l.legend(handles=handles, loc="upper right", fontsize=9)
    ax_l.add_artist(leg_colors)


    if n_traj > 2:
        step_lines = []
        for k, (ii, jj) in enumerate(unique_traj):
            tag = "start" if k == 0 else ("end" if k == n_traj - 1 else f"  {k}")
            step_lines.append(
                f"{tag}: ({int(ATTACK_GRID[ii]*100):>3d}%, "
                f"{int(ATTACK_GRID[jj]*100):>3d}%)"
            )
        step_text = "\n".join(step_lines)
        ax_l.text(0.02, 0.02, step_text, transform=ax_l.transAxes,
                  fontsize=8, family="monospace", verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                            edgecolor="gray", alpha=0.85))
    ax_l.set_xlabel("Utility of Location 0")
    ax_l.set_ylabel("Utility of Location 1")
    ax_l.set_title("(a) Best-response trajectory in utility space")
    ax_l.grid(True, alpha=0.3)

    # right panel: incentive to deviate from truthful
    lam_pct = ATTACK_GRID * 100
    dU_L0 = U0[:, 0] - U0[0, 0]   # L0 deviates alone
    dU_L1 = U1[0, :] - U1[0, 0]   # L1 deviates alone

    y_max = max(dU_L0.max(), dU_L1.max(), 0)
    y_min = min(dU_L0.min(), dU_L1.min(), 0)
    pad = 0.1 * max(abs(y_max), abs(y_min), 0.01)
    ax_r.axhspan(0, y_max + pad, alpha=0.08, color="green", zorder=0)
    ax_r.axhspan(y_min - pad, 0, alpha=0.08, color="red", zorder=0)

    ax_r.plot(lam_pct, dU_L0, "o-", color="tab:blue", lw=2.5, markersize=11,
              label="$L_0$ deviates  ($L_1$ truthful)", zorder=3)
    ax_r.plot(lam_pct, dU_L1, "s-", color="tab:orange", lw=2.5, markersize=11,
              label="$L_1$ deviates  ($L_0$ truthful)", zorder=3)

    ax_r.axhline(0, color="black", ls="-", lw=1.2, alpha=0.8, zorder=2)
    ax_r.text(lam_pct[-1], 0, "  truthful baseline", va="bottom", ha="right",
              fontsize=9, style="italic", color="black")

    br0 = int(np.argmax(dU_L0))
    br1 = int(np.argmax(dU_L1))
    ax_r.scatter([lam_pct[br0]], [dU_L0[br0]], s=350, marker="*",
                 color="tab:blue", edgecolor="black", linewidth=1.5,
                 zorder=6)
    ax_r.scatter([lam_pct[br1]], [dU_L1[br1]], s=350, marker="*",
                 color="tab:orange", edgecolor="black", linewidth=1.5,
                 zorder=6)

    verdict0 = "ATTACK pays" if dU_L0[br0] > 1e-6 else "truthful wins"
    verdict1 = "ATTACK pays" if dU_L1[br1] > 1e-6 else "truthful wins"

    K_local = len(lam_pct)
    if br0 >= K_local - 1:
        dx0, ha0 = -10, "right"
    else:
        dx0, ha0 = 10, "left"
    if br1 >= K_local - 1:
        dx1, ha1 = -10, "right"
    else:
        dx1, ha1 = 10, "left"

    ax_r.annotate(f"BR = {int(lam_pct[br0])}%  ({verdict0}: dU={dU_L0[br0]:+.3f})",
                  (lam_pct[br0], dU_L0[br0]),
                  textcoords="offset points", xytext=(dx0, 12),
                  ha=ha0, fontsize=9.5, color="tab:blue", fontweight="bold")
    ax_r.annotate(f"BR = {int(lam_pct[br1])}%  ({verdict1}: dU={dU_L1[br1]:+.3f})",
                  (lam_pct[br1], dU_L1[br1]),
                  textcoords="offset points", xytext=(dx1, -18),
                  ha=ha1, fontsize=9.5, color="tab:orange", fontweight="bold")

    ax_r.set_xlabel("Deviating location's attack level $\\lambda$ (%)")
    ax_r.set_ylabel("Utility shift vs truthful  "
                    "$\\Delta U_l = U_l(\\lambda,0) - U_l(0,0)$")
    ax_r.set_title("(b) Incentive to deviate from truthful strategy")
    ax_r.legend(loc="best", fontsize=10)
    ax_r.grid(True, alpha=0.3)
    ax_r.set_xticks(lam_pct)
    ax_r.set_xticklabels([f"{int(p)}%" for p in lam_pct])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Paper-style plot saved to {out_path}  "
          f"(BR_L0={int(lam_pct[br0])}% [{verdict0}], "
          f"BR_L1={int(lam_pct[br1])}% [{verdict1}])")


def plot_payoff_heatmaps(U0, U1, E, traj_async, label, out_path):
    # 4-panel figure:
    # (a) U_L0 heatmap
    # (b) U_L1 heatmap
    # (c) refugee employment rate heatmap
    # (d) (U_0, U_1) scatter + BR trajectory, color = employment rate
    K = len(ATTACK_GRID)
    tick_labels = [f"{int(x*100)}%" for x in ATTACK_GRID]
    traj_set = set(traj_async)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Location-attack game  —  {label} case", fontsize=14,
                 fontweight="bold")

    def draw_heatmap(ax, M, title, cmap="RdYlGn", fmt="{:+.2f}"):
        im = ax.imshow(M, cmap=cmap, origin="lower", aspect="auto")
        ax.set_xticks(range(K)); ax.set_xticklabels(tick_labels)
        ax.set_yticks(range(K)); ax.set_yticklabels(tick_labels)
        ax.set_xlabel("$\\lambda_1$ (Location 1 attack level)")
        ax.set_ylabel("$\\lambda_0$ (Location 0 attack level)")
        ax.set_title(title)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=9, color="black",
                        fontweight="bold" if (i, j) in traj_set else "normal")
        plt.colorbar(im, ax=ax, shrink=0.8)

    draw_heatmap(axes[0, 0], U0, "$U_{L_0}$  (Location 0's utility)")
    draw_heatmap(axes[0, 1], U1, "$U_{L_1}$  (Location 1's utility)")
    e_truthful = E[0, 0]
    draw_heatmap(axes[1, 0], E,
                 f"Refugee employment rate  (truthful = {e_truthful:.3f})",
                 fmt="{:.3f}")

    # panel 4
    ax = axes[1, 1]
    e_flat = E.flatten()
    sc = ax.scatter(U0.flatten(), U1.flatten(), c=e_flat,
                    cmap="RdYlGn", s=120, edgecolor="gray", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Refugee employment rate")

    for i in range(K):
        for j in range(K):
            ax.annotate(f"{int(ATTACK_GRID[i]*100)},{int(ATTACK_GRID[j]*100)}",
                        (U0[i, j], U1[i, j]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="dimgray")

    xs = [U0[i, j] for (i, j) in traj_async]
    ys = [U1[i, j] for (i, j) in traj_async]
    ax.plot(xs, ys, "-", color="tab:blue", linewidth=2, alpha=0.7,
            label="async BR path")
    for k in range(len(traj_async) - 1):
        i0, j0 = traj_async[k]
        i1, j1 = traj_async[k + 1]
        ax.annotate("",
                    xy=(U0[i1, j1], U1[i1, j1]),
                    xytext=(U0[i0, j0], U1[i0, j0]),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))

    i_start, j_start = traj_async[0]
    i_end, j_end = traj_async[-1]
    e_start = E[i_start, j_start]
    e_end = E[i_end, j_end]
    ax.scatter([U0[i_start, j_start]], [U1[i_start, j_start]], marker="o",
               s=250, facecolor="none", edgecolor="black", linewidth=2,
               label=f"start (0%, 0%)  —  emp={e_start:.3f}")
    ax.scatter([U0[i_end, j_end]], [U1[i_end, j_end]], marker="*",
               s=400, color="red", edgecolor="black", linewidth=1,
               label=f"NE ({int(ATTACK_GRID[i_end]*100)}%, "
                     f"{int(ATTACK_GRID[j_end]*100)}%)  —  "
                     f"emp={e_end:.3f}  "
                     f"(d={e_end - e_start:+.3f})",
               zorder=5)

    ax.set_xlabel("$U_{L_0}$  (Location 0 utility)")
    ax.set_ylabel("$U_{L_1}$  (Location 1 utility)")
    ax.set_title("(U_0, U_1) space — async BR dynamics")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {out_path}  "
          f"(truthful emp={e_start:.3f}, NE emp={e_end:.3f}, "
          f"d={e_end - e_start:+.3f})")



def plot_bounds_1d_slice(E_learned, E_oracle, E_strat, out_path,
                          attacker="L0"):
    # un seul location attaque, l'autre reste truthful.
    # ca fait ressortir le max gap entre strat-aware et true oracle parce
    # que strat-aware peut rerouter tous les types attaques vers la location
    # truthful, tandis que true oracle subit les attaques passivement.
    K = len(ATTACK_GRID)
    if attacker == "L0":
        # L1 truthful (j=0), L0 varie (i=0..K-1)
        e_l = np.array([E_learned[i, 0] for i in range(K)])
        e_o = np.array([E_oracle[i, 0]  for i in range(K)])
        e_s = np.array([E_strat[i, 0]   for i in range(K)])
        x = ATTACK_GRID * 100
        xlabel = "$\\lambda_0$  (Location 0 attack level, Location 1 truthful)"
    else:
        # L0 truthful (i=0), L1 varie (j=0..K-1)
        e_l = np.array([E_learned[0, j] for j in range(K)])
        e_o = np.array([E_oracle[0, j]  for j in range(K)])
        e_s = np.array([E_strat[0, j]   for j in range(K)])
        x = ATTACK_GRID * 100
        xlabel = "$\\lambda_1$  (Location 1 attack level, Location 0 truthful)"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, e_s, "^-",  color="tab:cyan",   lw=2.5, markersize=11,
            label="Strategy-aware oracle (knows V_true + $\\lambda$)")
    ax.plot(x, e_o, "s--", color="tab:orange", lw=2.2, markersize=10,
            label="True oracle (knows V_true only)")
    ax.plot(x, e_l, "o-",  color="tab:blue",   lw=2,   markersize=9,
            label="Learned predictor (noisy)")

    ax.fill_between(x, e_o, e_s, color="tab:cyan",   alpha=0.15,
                    label="strat-aware advantage")
    ax.fill_between(x, e_l, e_o, color="tab:orange", alpha=0.12,
                    label="learning noise")

    ax.axhline(e_s[0], color="tab:cyan", ls=":", lw=1, alpha=0.6)
    ax.annotate(f"  truthful baseline = {e_s[0]:.3f}",
                (x[-1], e_s[0]), textcoords="offset points", xytext=(-100, 5),
                fontsize=9, color="tab:cyan", style="italic")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Refugee employment rate  (avg over steady-state rounds)")
    ax.set_title(f"Employment rate bounds ({attacker} attacks alone)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)}%" for p in x])
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  1D slice plot saved to {out_path}  "
          f"(max gap strat-oracle = {(e_s - e_o).max():+.3f} at "
          f"lambda={int(x[(e_s - e_o).argmax()])}%)")


def run_full_analysis():

    print("Game-theoretic analysis of location attacks (MPC version)")
    print(f"  attack grid: {[int(x*100) for x in ATTACK_GRID]}%")
    print(f"  n_seeds = {N_SEEDS}, n_rounds = {N_ROUNDS}, "
          f"MPC horizon = {MPC_HORIZON} rounds")


    print(f"({len(ATTACK_GRID)**2} scenarios)...")
    (U0L, U1L, U0O, U1O, U0S, U1S,
     EL, EO, ES) = compute_payoff_matrix(use_cache=True)

    print("\n[2] Payoff matrices")
    print_payoff_matrix(U0L, U1L, "LEARNED case: (U_L0, U_L1)")
    print_payoff_matrix(U0O, U1O, "TRUE ORACLE case (knows V_true, not lambda): (U_L0, U_L1)")
    print_payoff_matrix(U0S, U1S, "STRAT-AWARE ORACLE (knows V_true + prev lambda): (U_L0, U_L1)")

    print("\n[3] Binary scenarios (truthful vs full attack)")
    binary_scenario_report(U0L, U1L, "learned")
    binary_scenario_report(U0O, U1O, "oracle")
    binary_scenario_report(U0S, U1S, "strat-aware")

    print("\n[4] Equilibrium analysis")
    report_equilibria(U0L, U1L, "learned")
    report_equilibria(U0O, U1O, "oracle")
    report_equilibria(U0S, U1S, "strat-aware")

    print("\n[5] Best-response trajectories (async)")

    traj_L = best_response_async(U0L, U1L, noise_sigma=0.008, noise_seed=41)
    traj_O = best_response_async(U0O, U1O)
    traj_S = best_response_async(U0S, U1S)
    for label, traj in [("learned", traj_L), ("oracle", traj_O),
                        ("strat-aware", traj_S)]:
        steps = [f"({int(ATTACK_GRID[i]*100)}%,{int(ATTACK_GRID[j]*100)}%)"
                 for (i, j) in traj]
        print(f"  {label:12s}: {' -> '.join(steps)}")

    print("\n[6] Paper-style plots (BR trajectory + incentive to deviate)")
    plot_paper_style(U0L, U1L, traj_L, "learned",     "BR_learned.png")
    plot_paper_style(U0O, U1O, traj_O, "true oracle", "outcomeaware_oracle.png")
    plot_paper_style(U0S, U1S, traj_S, "strat-aware", "strataware_oracle.png")

    print("\n[7] 4-panel heatmap plots (U_L0, U_L1, employment, trajectory)")
    plot_payoff_heatmaps(U0L, U1L, EL, traj_L, "learned",
                         "heatmap_learned.png")
    plot_payoff_heatmaps(U0O, U1O, EO, traj_O, "true oracle",
                         "heatmap_oracle.png")
    plot_payoff_heatmaps(U0S, U1S, ES, traj_S, "strat-aware",
                         "heatmap_strat_oracle.png")

    print("\n[8] Bounds plot: 1D slice (asymmetric attacks reveal strat-aware gap)")
    plot_bounds_1d_slice(EL, EO, ES, "bounds_L0attacks.png",
                         attacker="L0")
    plot_bounds_1d_slice(EL, EO, ES, "bounds_L1attacks.png",
                         attacker="L1")


if __name__ == "__main__":
    patch_globals()
    run_full_analysis()
