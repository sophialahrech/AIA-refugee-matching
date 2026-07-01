# Asynchronous IBR over all 16 locations (appendix).
# Every location attacks and updates its lambda in turn, seeing the others'
# latest choices. We check it converges to the same equilibrium from both the
# all-truthful and the all-attack start.

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import (N_TYPES, TYPE_PROBS, round_assignment, location_attacks,
                        C_APPENDIX, LocPredictor, solve_lp, per_loc_utility)
from play_multi_loc_attack import build_full_model, LOCS_ALL

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
})

HERE        = os.path.dirname(__file__)
BETA        = 0.9
C           = C_APPENDIX
N_SEEDS     = 12
ATTACK_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
PLANNER     = "strat"
MAX_SWEEPS  = 8
CACHE       = os.path.join(HERE, "async_ibr_16loc.npz")


def simulate_profile_utility(model, atk_vec, n_seeds, beta=BETA, c=C):
    arr       = model["quarterly_arrivals"]
    T         = len(arr)
    n_loc     = len(model["locs"])
    theta     = model["theta"]
    cost_mask = model["cost_mask"]
    alpha     = model["alpha"]
    weights   = beta ** np.arange(T)

    U = np.zeros((n_seeds, n_loc))
    for s in range(n_seeds):
        np.random.seed(s)
        if PLANNER == "learned":
            pred = LocPredictor(n_loc, random_state=s)
            tb, lb, ob = [], [], []
        for t in range(T):
            n_t = int(arr[t])
            if n_t == 0:
                continue
            types  = np.random.choice(N_TYPES, size=n_t, p=TYPE_PROBS)
            V_true = theta[types]
            lambdas = location_attacks(types, atk_vec, cost_mask)
            if PLANNER == "strat":
                V = (1.0 - lambdas) * V_true
            else:
                V = pred.predict(types)
            a = round_assignment(solve_lp(V, alpha)).argmax(axis=1)
            out = np.array([np.random.binomial(
                1, (1 - lambdas[i, a[i]]) * V_true[i, a[i]]) for i in range(n_t)])
            if PLANNER == "learned":
                tb.append(types); lb.append(a); ob.append(out)
                pred.train(np.concatenate(tb), np.concatenate(lb),
                           np.concatenate(ob))
            u = per_loc_utility(types, a, cost_mask, c=c, beta_indiv=beta,
                                theta=theta, attack_levels_vec=atk_vec)
            U[s] += weights[t] * u
    return U.mean(axis=0)


def async_ibr(model, start="truthful", verbose=True):
    """Each location best-responds in turn until no one moves.
    Returns the history of attack vectors and the final equilibrium."""
    n_loc = len(model["locs"])
    if start == "truthful":
        lam = np.zeros(n_loc)
    elif start == "full":
        lam = np.ones(n_loc)
    else:
        lam = np.array(start, dtype=float)

    history = [lam.copy()]
    for sweep in range(MAX_SWEEPS):
        changed = False
        for l in range(n_loc):
            best_val, best_lam = -np.inf, lam[l]
            for cand in ATTACK_GRID:
                trial = lam.copy(); trial[l] = cand
                U = simulate_profile_utility(model, trial, N_SEEDS)
                if U[l] > best_val:
                    best_val, best_lam = U[l], cand
            if best_lam != lam[l]:
                changed = True
            lam[l] = best_lam
            history.append(lam.copy())
        if verbose:
            print(f"  [{start}] after sweep {sweep+1}: "
                  f"lambda = {np.round(lam,2).tolist()}")
        if not changed:
            if verbose:
                print(f"  [{start}] converged after sweep {sweep+1}")
            break
    return history, lam.copy()


def main():
    t0 = time.time()
    model = build_full_model(LOCS_ALL)
    n_loc = len(model["locs"])
    print(f"Async IBR over {n_loc} locations, planner={PLANNER}, "
          f"n_seeds={N_SEEDS}, beta={BETA}")

    print("\n== Run A: start from truthful (all 0) ==")
    hist_T, eq_T = async_ibr(model, start="truthful")
    print("\n== Run B: start from full attack (all 1) ==")
    hist_F, eq_F = async_ibr(model, start="full")

    same = np.allclose(eq_T, eq_F)
    print(f"\nEquilibrium from truthful start: {np.round(eq_T,2).tolist()}")
    print(f"Equilibrium from full start:     {np.round(eq_F,2).tolist()}")
    print(f"Same equilibrium from both starts: {same}")

    n_att = int((eq_T > 0).sum())
    print(f"\n#locations attacking at equilibrium: {n_att}/{n_loc}")

    np.savez(CACHE,
             hist_truthful=np.array(hist_T),
             hist_full=np.array(hist_F),
             eq_truthful=eq_T, eq_full=eq_F,
             locs=np.array(model["locs"]),
             grid=ATTACK_GRID, beta=BETA, n_seeds=N_SEEDS)

    plot_trajectory(model, hist_T, hist_F, eq_T)
    print(f"\nDone in {time.time()-t0:.1f}s")


def plot_trajectory(model, hist_T, hist_F, eq):
    locs = model["locs"]
    n_loc = len(locs)
    H_T = np.array(hist_T)
    H_F = np.array(hist_F)
    cmap = plt.cm.tab20

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2),
                                   gridspec_kw={"wspace": 0.18})
    for ax, H, title in [(axA, H_T, "(a)  start: all truthful  $\\lambda=0$"),
                         (axB, H_F, "(b)  start: full attack  $\\lambda=1$")]:
        steps = np.arange(H.shape[0])
        for l in range(n_loc):
            # small vertical offset so overlapping lines stay readable
            ax.plot(steps, H[:, l] + 0.012 * (l - n_loc / 2) / n_loc,
                    marker="o", markersize=3, lw=1.4, color=cmap(l % 20),
                    label=f"Loc {locs[l]}")
        for sw in range(0, H.shape[0], n_loc):
            ax.axvline(sw, color="gray", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("asynchronous update step\n(one location updates per step)")
        ax.set_ylabel(r"attack level $\lambda_\ell$")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks(ATTACK_GRID)
        ax.grid(alpha=0.25)
        ax.set_title(title, loc="left")
    axB.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1,
               fontsize=8, framealpha=0.95)
    fig.suptitle("Asynchronous IBR over all 16 locations converges to the same "
                 "equilibrium from both starting profiles", fontweight="bold")
    out = os.path.join(HERE, "async_ibr_16loc_trajectory.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
