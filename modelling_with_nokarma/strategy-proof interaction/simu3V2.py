# game-theoretic analysis - IBR version (iterative best response a la
# Ionescu et al. 2023, school choice paper).
#
# Difference from simu3 (MPC): here each location is bounded-rational.
# It does NOT see the full row/column of the payoff matrix at once.
# It only compares its current lambda with its neighboring values
# (+/- 1 step) on the attack grid. It moves up or down by one step
# if that yields a higher payoff. As a result, it does not have the
# global view of "utility is monotone, so jump directly to 100%" 

import os
import numpy as np
import matplotlib.pyplot as plt

import simu2
import simu3   


N_SEEDS = simu3.N_SEEDS
N_ROUNDS = simu3.N_ROUNDS
ATTACK_GRID = simu3.ATTACK_GRID


def patch_globals():
    simu2.n_seeds = N_SEEDS
    simu2.n_rounds = N_ROUNDS



def best_response_ibr_async(U0, U1, max_iters=40):
    K = U0.shape[0]
    i, j = 0, 0
    traj = [(i, j)]
    seen = {(i, j)}
    for t in range(max_iters):
        cand_i = [c for c in (i - 1, i, i + 1) if 0 <= c < K]
        i_new = max(cand_i, key=lambda c: U0[c, j])
        changed_a = (i_new != i)
        i = i_new
        traj.append((i, j))
        if (i, j) in seen and changed_a:
            break
        seen.add((i, j))

        cand_j = [c for c in (j - 1, j, j + 1) if 0 <= c < K]
        j_new = max(cand_j, key=lambda c: U1[i, c])
        changed_b = (j_new != j)
        j = j_new
        traj.append((i, j))
        if (i, j) in seen and changed_b:
            break
        seen.add((i, j))

        if not changed_a and not changed_b:
            break
    return traj




def run_full_analysis_ibr():
    print("Game-theoretic analysis of location attacks (IBR version)")
    print(f"  attack grid: {[int(x*100) for x in ATTACK_GRID]}%")
    print(f"  n_seeds = {N_SEEDS}, n_rounds = {N_ROUNDS}")


    print("\n Loading / computing 5x5 payoff matrix (shared with simu3)...")
    (U0L, U1L, _, _, _, _, EL, _, _) = simu3.compute_payoff_matrix(use_cache=True)

    print("\n[2] Payoff matrix (learned case)")
    simu3.print_payoff_matrix(U0L, U1L, "LEARNED case: (U_L0, U_L1)")

    print("\n[3] Binary-attack shifts (learned)")
    simu3.binary_scenario_report(U0L, U1L, "learned")

    print("\n[4] Equilibrium analysis (learned)")
    simu3.report_equilibria(U0L, U1L, "learned")

    print("\n[5] IBR iteration (sync vs async, local +/-1 moves) -- learned")
    traj_ibr = best_response_ibr_async(U0L, U1L)
    print(f"  IBR trajectory: {traj_ibr}")

    out_dir = os.path.dirname(__file__)
    paper_out = os.path.join(out_dir, "BR_ibr_learned.png")
    heatmap_out = os.path.join(out_dir, "heatmap_ibr_learned.png")

    simu3.plot_paper_style(U0L, U1L, traj_ibr, "learned (IBR local +/-1)",
                           paper_out)
    simu3.plot_payoff_heatmaps(U0L, U1L, EL, traj_ibr,
                               "learned (IBR local +/-1)", heatmap_out)

    print(f"  PNG IBR saved to {paper_out}")
    print(f"  PNG heatmap IBR saved to {heatmap_out}")

if __name__ == "__main__":
    patch_globals()
    run_full_analysis_ibr()
