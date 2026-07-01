
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html


import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression


N_TYPES = 8

_REPL_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                         "EJ_matching_with_semibandits_replication_files-master")
THETA_CSV = os.path.join(_REPL_DIR, "Data_processed", "theta_calibrated.csv")
AGG_CSV   = os.path.join(_REPL_DIR, "Figures", "Aggregate_bytype.csv")
RAW_XLSX  = os.path.join(os.path.dirname(__file__), "..", "..",
                         "rawdata_analysed.xlsx")

TYPE_PROBS = pd.read_csv(AGG_CSV).sort_values("U")["n_total"].to_numpy(float)
TYPE_PROBS = TYPE_PROBS / TYPE_PROBS.sum()



def build_2loc_model(loc_a, loc_b):
    """Load real data for the two-location game restricted to locations
    `loc_a` and `loc_b`."""
    locs = [loc_a, loc_b]
    n_loc = 2

    df = pd.read_csv(THETA_CSV)
    theta = np.zeros((N_TYPES, n_loc))
    for l, v in enumerate(locs):
        sub = df[df["V"] == v].sort_values("U") #V is location, U is type, theta is success prob
        theta[:, l] = sub["theta"].values
    loc_summary = pd.read_excel(RAW_XLSX, sheet_name="03 Location summary")
    cap_per_loc = (loc_summary.set_index("Location V")
                   ["total_quarterly_capacity"].fillna(0.0))
    cap = np.array([float(cap_per_loc.loc[v]) for v in locs]) # total quarterly capacity per location
    alpha = cap / cap.sum()

  
    cap_by_q = pd.read_excel(RAW_XLSX, sheet_name="10 Capacity by quarter") #arrivals per quarter per location = capacity by quarter
    cols = [f"Loc {v}" for v in locs]
    quarterly = cap_by_q[cols].fillna(0).astype(int).sum(axis=1).values


    cost_mask = np.zeros((N_TYPES, n_loc), dtype=bool) #which types are "costly" for each location ( attacked when lambda > 0)
    for l in range(n_loc):
        med = np.median(theta[:, l])
        cost_mask[:, l] = theta[:, l] < med

    return dict(theta=theta, cap=cap, alpha=alpha, cost_mask=cost_mask,
                locs=locs, quarterly_arrivals=quarterly)


class LocPredictor:
    def __init__(self, n_loc, random_state=42):
        self.n_loc = n_loc
        self.models = [LogisticRegression(solver="liblinear", warm_start=True,
                                          max_iter=1000,
                                          random_state=random_state)
                       for _ in range(n_loc)]
        self.fitted = [False] * n_loc

    def _onehot(self, types):
        X = np.zeros((len(types), N_TYPES))
        X[np.arange(len(types)), types] = 1.0
        return X

    def train(self, types_hist, loc_hist, out_hist):
        X = self._onehot(types_hist)
        for l in range(self.n_loc):
            mask = (loc_hist == l)
            if mask.sum() < 2 or len(np.unique(out_hist[mask])) < 2:
                continue
            self.models[l].fit(X[mask], out_hist[mask])
            self.fitted[l] = True

    def predict(self, types):
        X = self._onehot(types)
        V = np.full((len(types), self.n_loc), 0.5)
        for l in range(self.n_loc):
            if self.fitted[l]:
                V[:, l] = self.models[l].predict_proba(X)[:, 1]
        return V



def solve_lp(V_for_lp, alpha):
    """Maximize sum_{i,l} V_for_lp[i,l] * x[i,l]  s.t.
       sum_l x[i,l] = 1 for all i,
       sum_i x[i,l] <= ceil(alpha[l] * r) for all l,
       x in [0,1]."""
    r, L = V_for_lp.shape
    V_for_lp = V_for_lp + np.random.uniform(0, 1e-8, size=V_for_lp.shape) # break ties randomly
    c = (-V_for_lp).flatten()   # minimize -V = maximize V
    
    # 1 row per refugee: its L cols sum to 1
    A_eq = np.zeros((r, r * L))
    for i in range(r):
        A_eq[i, i * L:(i + 1) * L] = 1.0  # refugee i's own L cols
    b_eq = np.ones(r)

    cap = np.ceil(alpha * r).astype(float)  # slots/loc this quarter 

    # 1 row per loc: its refugees (cols i*L+l) sum <= cap
    A_ub = np.zeros((L, r * L))
    for l in range(L):
        for i in range(r):
            A_ub[l, i * L + l] = 1.0 # cells spaced by L

    res = linprog(c, A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, 1)] * (r * L), method="highs")
    if not res.success:
        x = np.zeros((r, L))
        for i in range(r):
            x[i, np.random.randint(L)] = 1.0
        return x
    return res.x.reshape(r, L)


def round_assignment(x_frac):
    r, L = x_frac.shape
    out = np.zeros_like(x_frac)
    for i in range(r):
        row = x_frac[i]
        tied = np.flatnonzero(row >= row.max() - 1e-9)
        choice = tied[0] if len(tied) == 1 else np.random.choice(tied)
        out[i, choice] = 1.0
    return out

def location_attacks(types, attack_levels_vec, cost_mask):
    r = len(types)
    L = cost_mask.shape[1]
    lambdas = np.zeros((r, L))
    for l in range(L):
        attacked = cost_mask[types, l] # is this refugee type costly for this location?
        lambdas[:, l] = np.where(attacked, attack_levels_vec[l], 0.0)
    return lambdas



C_APPENDIX = 1.0   # support cost per quarter


def per_loc_utility(types, assigned, cost_mask, c=C_APPENDIX,
                    beta_indiv=1.0, theta=None, attack_levels_vec=None):
    """Per-location utility (Proposition A.1 expected discounted utility),
    summed over the refugees assigned to each location."""
    L = cost_mask.shape[1]
    util = np.zeros(L)
    for i in range(len(types)):
        l = assigned[i]
        k = types[i]
        lam_i = attack_levels_vec[l] * (1.0 if cost_mask[k, l] else 0.0)
        p = theta[k, l]
        denom = 1.0 - beta_indiv + p * beta_indiv
        if abs(denom) < 1e-12:
            denom = 1e-12
        util[l] += -c * (1.0 + beta_indiv * lam_i * p) / denom
    return util


def run_one_seed(seed, model, attack_levels_vec):
    """One full run (all quarters) for a given seed and attack profile.
    Runs 3 planners (learned / oracle / strat) on the same stream and
    returns their per-round employment (total and per type)."""
    np.random.seed(seed)
    arrivals = model["quarterly_arrivals"]
    n_rounds = len(arrivals)
    n_loc = len(model["locs"])
    cost_mask = model["cost_mask"]
    alpha = model["alpha"]

    predictor = LocPredictor(n_loc, random_state=seed)
    types_buf, locb, outb = [], [], []

    emp_L = np.zeros(n_rounds); emp_O = np.zeros(n_rounds); emp_S = np.zeros(n_rounds)
    emp_S_per_type = np.zeros(N_TYPES)
    cnt_per_type = np.zeros(N_TYPES)

    for t in range(n_rounds):
        n_t = int(arrivals[t])
        if n_t == 0:
            continue
        types = np.random.choice(N_TYPES, size=n_t, p=TYPE_PROBS)
        V_true = model["theta"][types]   # baseline success prob per individual and location
        lambdas = location_attacks(types, attack_levels_vec, cost_mask)

        # learned planner: trains on observed outcomes
        V_hat = predictor.predict(types)
        a_L = round_assignment(solve_lp(V_hat, alpha)).argmax(axis=1)
        out_L = np.array([np.random.binomial(1, (1 - lambdas[i, a_L[i]]) * V_true[i, a_L[i]])
                          for i in range(n_t)])
        types_buf.append(types); locb.append(a_L); outb.append(out_L)
        predictor.train(np.concatenate(types_buf),
                        np.concatenate(locb),
                        np.concatenate(outb))

        # true oracle: knows V_true, ignores attacks
        a_O = round_assignment(solve_lp(V_true, alpha)).argmax(axis=1)
        out_O = np.array([np.random.binomial(1, (1 - lambdas[i, a_O[i]]) * V_true[i, a_O[i]])
                          for i in range(n_t)])

        # strategy-aware oracle: knows V_true and lambda, reroutes
        V_strat = (1 - lambdas) * V_true
        a_S = round_assignment(solve_lp(V_strat, alpha)).argmax(axis=1)
        out_S = np.array([np.random.binomial(1, (1 - lambdas[i, a_S[i]]) * V_true[i, a_S[i]])
                          for i in range(n_t)])

        emp_L[t] = out_L.mean(); emp_O[t] = out_O.mean(); emp_S[t] = out_S.mean()
        for k in range(N_TYPES):
            mask = (types == k)
            cnt_per_type[k] += mask.sum()
            emp_S_per_type[k] += out_S[mask].sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        emp_S_rate_per_type = np.where(cnt_per_type > 0,
                                       emp_S_per_type / np.maximum(cnt_per_type, 1),
                                       np.nan)

    return dict(emp_L=emp_L, emp_O=emp_O, emp_S=emp_S,
                emp_S_per_type=emp_S_rate_per_type,
                cnt_per_type=cnt_per_type)


ATTACK_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.00])


def find_pure_nash(U0, U1):
    K = U0.shape[0]
    return [(i, j) for i in range(K) for j in range(K)
            if U0[i, j] == U0[:, j].max() and U1[i, j] == U1[i, :].max()]
