# fine-grid scan of the strat-aware payoff matrix at 10% increments.
# goal: verify the threshold conjecture -> U_L1 is ~flat for lambda_1 <= lambda_0
import os
import numpy as np
import matplotlib.pyplot as plt

import simu2
import simu3


FINE_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 3)  
N_SEEDS_FINE = 10
CACHE_FINE = "payoff_matrix_cache_fine11.npz"


def patch_globals():
    simu2.n_seeds = N_SEEDS_FINE
    simu2.n_rounds = simu3.N_ROUNDS


def compute_fine_matrix(use_cache=True):
    K = len(FINE_GRID)
    if use_cache and os.path.exists(CACHE_FINE):
        d = np.load(CACHE_FINE)
        if (d["grid"].shape == FINE_GRID.shape
                and np.allclose(d["grid"], FINE_GRID)
                and int(d["n_seeds"]) == N_SEEDS_FINE):
            print(f"  Loaded cached fine-grid matrix from {CACHE_FINE}")
            return d["U0_strat"], d["U1_strat"], d["E_strat"]
        print(f"  Cache at {CACHE_FINE} is stale, recomputing.")

    U0_strat = np.zeros((K, K))
    U1_strat = np.zeros((K, K))
    E_strat  = np.zeros((K, K))

    total = K * K
    k = 0
    for i, lam0 in enumerate(FINE_GRID):
        for j, lam1 in enumerate(FINE_GRID):
            k += 1
            print(f"  [{k:3d}/{total}] lambda = ({lam0:.2f}, {lam1:.2f}) ...",
                  end="", flush=True)
            ul, uo, us, el, eo, es = simu3.run_scenario(
                lam0, lam1, n_seeds=N_SEEDS_FINE
            )
            U0_strat[i, j] = us[:, 0].mean()
            U1_strat[i, j] = us[:, 1].mean()
            E_strat[i, j]  = es.mean()
            print(f"  U_L0={U0_strat[i,j]:+.3f}  U_L1={U1_strat[i,j]:+.3f}  "
                  f"E={E_strat[i,j]:.3f}")

    np.savez(CACHE_FINE, grid=FINE_GRID,
             n_seeds=N_SEEDS_FINE, n_rounds=simu3.N_ROUNDS,
             U0_strat=U0_strat, U1_strat=U1_strat, E_strat=E_strat)
    print(f"  Cached fine-grid matrix to {CACHE_FINE}")
    return U0_strat, U1_strat, E_strat


def plot_u1_heatmap_fine(U1, out_path):
    K = len(FINE_GRID)
    tick_labels = [f"{int(x*100)}%" for x in FINE_GRID]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(U1, cmap="RdYlGn", origin="lower", aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticks(range(K))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(r"$\lambda_1$ (Location 1 attack level)")
    ax.set_ylabel(r"$\lambda_0$ (Location 0 attack level)")
    ax.set_title(r"$U_{L_1}$  (Location 1's utility)  —  strat-aware, 10% grid  "
                 f"[n_seeds={N_SEEDS_FINE}]",
                 fontweight="bold")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{U1[i,j]:+.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")

    diag = np.arange(K)
    ax.plot(diag, diag, color="black", linestyle="--", linewidth=1.8,
            alpha=0.6, label=r"$\lambda_1 = \lambda_0$ (conjectured threshold)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.colorbar(im, ax=ax, shrink=0.8, label=r"$U_{L_1}$")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved to {out_path}")




def threshold_report(U1):
    
    K = len(FINE_GRID)
    print("\n  === Threshold verification ===")
    print("  For each row (lambda_0), we expect:")
    print("    - below diagonal (lambda_1 < lambda_0): ~flat, slight decrease")
    print("    - above diagonal (lambda_1 > lambda_0): step up")
    print()
    print(f"  {'lam0':>5} | {'below_spread':>14} {'above_mean':>12} "
          f"{'below_mean':>12} {'jump_at_diag':>14}")
    print("  " + "-" * 70)
    for i in range(1, K - 1): 
        below = U1[i, :i]       # lambda_1 < lambda_0
        above = U1[i, i+1:]     # lambda_1 > lambda_0
        bspread = below.max() - below.min() if len(below) > 0 else float("nan")
        bmean   = below.mean()  if len(below) > 0 else float("nan")
        amean   = above.mean()  if len(above) > 0 else float("nan")
        jump    = U1[i, min(i+1, K-1)] - U1[i, max(i-1, 0)]
        print(f"  {int(FINE_GRID[i]*100):3d}% | "
              f"{bspread:>+14.3f} {amean:>+12.3f} {bmean:>+12.3f} "
              f"{jump:>+14.3f}")


if __name__ == "__main__":
    print("=" * 90)
    print("Fine-grid scan of strat-aware payoff matrix (11x11, 10% increments)")
    print(f"  n_seeds = {N_SEEDS_FINE}, n_rounds = {simu3.N_ROUNDS}")
    print(f"  total cells = {len(FINE_GRID)**2}")
    print("=" * 90)

    patch_globals()
    U0, U1, E = compute_fine_matrix(use_cache=True)
    plot_u1_heatmap_fine(U1, "heatmap_U1_fine_grid.png")
    threshold_report(U1)
