# Multi-location HIAS model + assignment LP with a minimum-intake floor.
# Helper module imported by play_multi_loc_step1 / step2 / step3 / ci / step2_mpc.

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import N_TYPES, THETA_CSV, RAW_XLSX


# 16 HIAS locations (Loc 11 excluded: zero capacity).
LOCS_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]


# Multi-location model loader.
def build_full_model(locs):
    """Load real HIAS data for the given list of locations.
    Returns dict {theta, cap, alpha, cost_mask, locs, quarterly_arrivals}.
    """
    n_loc = len(locs)
    df = pd.read_csv(THETA_CSV)
    theta = np.zeros((N_TYPES, n_loc))
    for l, v in enumerate(locs):
        sub = df[df["V"] == v].sort_values("U")
        theta[:, l] = sub["theta"].values

    loc_summary = pd.read_excel(RAW_XLSX, sheet_name="03 Location summary")
    cap_per_loc = (loc_summary.set_index("Location V")
                   ["total_quarterly_capacity"].fillna(0.0))
    cap = np.array([float(cap_per_loc.loc[v]) for v in locs])
    alpha = cap / cap.sum()

    cap_by_q = pd.read_excel(RAW_XLSX, sheet_name="10 Capacity by quarter")
    cols = [f"Loc {v}" for v in locs]
    quarterly = cap_by_q[cols].fillna(0).astype(int).sum(axis=1).values

    cost_mask = np.zeros((N_TYPES, n_loc), dtype=bool)
    for l in range(n_loc):
        med = np.median(theta[:, l])
        cost_mask[:, l] = theta[:, l] < med

    return dict(theta=theta, cap=cap, alpha=alpha, cost_mask=cost_mask,
                locs=locs, quarterly_arrivals=quarterly)


# LP with both an upper cap and a minimum-intake floor.
def solve_lp_with_kappa(V_for_lp, cap_abs, kappa):
    """Solve the assignment LP with a per-location minimum-intake floor and
    absolute upper cap (not prop to arrivals).

        max  sum_{i,l} V[i,l] * x[i,l]
        s.t. sum_l x[i,l] = 1               for all i
             sum_i x[i,l] <= cap_abs[l]      for all l
             sum_i x[i,l] >= kappa[l]        for all l
             0 <= x[i,l] <= 1
    """
    r, L = V_for_lp.shape
    V_for_lp = V_for_lp + np.random.uniform(0, 1e-8, size=V_for_lp.shape)
    c = (-V_for_lp).flatten()
    A_eq = np.zeros((r, r * L))
    for i in range(r):
        A_eq[i, i * L:(i + 1) * L] = 1.0
    b_eq = np.ones(r)

    # cannot send more than r refugees to one location
    cap_max = np.minimum(cap_abs, r).astype(float)
    A_ub = np.zeros((2 * L, r * L))
    b_ub = np.zeros(2 * L)
    for l in range(L):
        for i in range(r):
            A_ub[l, i * L + l] = 1.0           # upper cap row
            A_ub[L + l, i * L + l] = -1.0      # floor row
        b_ub[l]     = cap_max[l]
        b_ub[L + l] = -kappa[l]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, 1)] * (r * L), method="highs")
    return res.x.reshape(r, L)
