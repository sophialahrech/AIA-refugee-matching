#IBR helpers used by the figure scripts.

import numpy as np
from simu2_real import run_one_seed, N_TYPES

TYPE_NAMES = [
    "F, no-Eng, not-WA",   # 0
    "M, no-Eng, not-WA",   # 1
    "F, Eng, not-WA",      # 2
    "M, Eng, not-WA",      # 3
    "F, no-Eng, WA",       # 4
    "M, no-Eng, WA",       # 5
    "F, Eng, WA",          # 6
    "M, Eng, WA",          # 7
]

N_SEEDS_TRAJ = 20


def best_response_async(U0, U1, max_iters=20, start=(0, 0)):
    """Async best response on the heatmap. L0 moves first (argmax over its
    column), then L1, until a fixed point.
    Returns (trajectory, cycle_start or None, status)."""
    i, j = start #both are truthfull at the strat 
    traj = [(i, j)]
    for _ in range(max_iters):
        prev = (i, j)
        i_new = int(np.argmax(U0[:, j])) # L0 best response to current j
        if i_new != i:
            i = i_new
            if (i, j) in traj:
                traj.append((i, j))
                return traj, traj.index((i, j)), "cycle"
            traj.append((i, j))
        j_new = int(np.argmax(U1[i, :]))   # L1 best response to current i
        if j_new != j:
            j = j_new
            if (i, j) in traj[:-1]:
                traj.append((i, j))
                return traj, traj.index((i, j)), "cycle"
            traj.append((i, j))
        if (i, j) == prev:
            return traj, None, "fixed"
    return traj, None, "diverged"


def simulate_cell(model, lam_vec, n_seeds):
    """Total and per-type employment at one (lambda_0, lambda_1) cell, using the strat-aware planner."""
    emps = np.zeros(n_seeds) # total employment per seed
    per_type = np.zeros((n_seeds, N_TYPES)) # per-type employment per seed 
    cnts = np.zeros((n_seeds, N_TYPES)) # how many of each type appeared per seed 
    for s in range(n_seeds):
        r = run_one_seed(s, model, lam_vec)
        emps[s] = r["emp_S"].mean()  
        per_type[s] = r["emp_S_per_type"]
        cnts[s] = r["cnt_per_type"]
   
    pt_mean = np.full(N_TYPES, np.nan) # per-type mean across seeds, weighted by how often each type appeared
    pt_std = np.full(N_TYPES, np.nan) #
    for k in range(N_TYPES): #for each type,  compute the average employment across seeds
        valid = ~np.isnan(per_type[:, k])
        w = cnts[valid, k]
        if valid.any() and w.sum() > 0:
            pt_mean[k] = np.average(per_type[valid, k], weights=w)
            pt_std[k] = float(np.std(per_type[valid, k]))
    return float(emps.mean()), pt_mean, float(emps.std()), pt_std
