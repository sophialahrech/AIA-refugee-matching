# baseline simulation, pas d'attaques
# predictor = logistic regression per location on 8 types 

import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


THETA_CSV = os.path.join(
    "..", "..", "EJ_matching_with_semibandits_replication_files-master",
    "Data_processed", "theta_calibrated.csv",
)


locations = [12, 14]
n_locations = len(locations)
n_types = 8

type_names = [
    "F, no-Eng, not-WA",   
    "M, no-Eng, not-WA",
    "F, Eng,    not-WA",
    "M, Eng,    not-WA",
    "F, no-Eng, WA",
    "M, no-Eng, WA",
    "F, Eng,    WA",
    "M, Eng,    WA",    
]

# empirical type distribution (from Aggregate_bytype.csv)
type_probs = np.array(
    [6048, 6272, 2816, 5632, 12128, 19264, 5344, 19424], dtype=float
)
type_probs = type_probs / type_probs.sum()

n_rounds = 50
batch_min = 20
batch_max = 100
alpha = np.array([0.6, 0.4])   # capacity shares
n_seeds = 50


def load_theta(locations):
    df = pd.read_csv(THETA_CSV)
    theta = np.zeros((n_types, len(locations)))
    for l, v in enumerate(locations):
        sub = df[df["V"] == v].sort_values("U")
        if len(sub) != n_types:
            raise ValueError(f"Affiliate V={v} does not have {n_types} U rows.")
        theta[:, l] = sub["theta"].values
    return theta


def generate_refugees(r):
    return np.random.choice(n_types, size=r, p=type_probs)


class TrueModel:
    # ground truth employment proba theta[u,l]

    def __init__(self, verbose=False):
        self.theta = load_theta(locations)
        if verbose:
            print(f"Loaded theta for affiliates {locations}:")
            for u in range(n_types):
                print(f"  U={u+1} ({type_names[u]}): {self.theta[u]}")

    def get_V(self, types):
        return self.theta[types]


def one_hot(types):
    r = len(types)
    X = np.zeros((r, n_types))
    X[np.arange(r), types] = 1.0
    return X


class LinearPredictor:
    # one LR per location

    def __init__(self, random_state=42):
        self.models = []
        for l in range(n_locations):
            clf = LogisticRegression(solver="liblinear", warm_start=True,
                                     max_iter=1000, random_state=random_state)
            self.models.append(clf)
        self.fitted = [False] * n_locations

    def train(self, history_types, history_l, history_o):
        X_all = one_hot(history_types)
        for l in range(n_locations):
            mask = history_l == l
            if mask.sum() < 2:
                continue
            X_l = X_all[mask]
            y_l = history_o[mask]
            if len(np.unique(y_l)) < 2:
                continue
            self.models[l].fit(X_l, y_l)
            self.fitted[l] = True

    def predict_V(self, types):
        r = len(types)
        X = one_hot(types)
        V_hat = np.full((r, n_locations), 0.5)
        for l in range(n_locations):
            if self.fitted[l]:
                V_hat[:, l] = self.models[l].predict_proba(X)[:, 1]
        return V_hat


def solve_lp(V_hat, alpha):
    r, L = V_hat.shape
    c = -V_hat.flatten()

    # chaque refugee -> 1 location
    assign_matrix = np.zeros((r, r * L))
    for i in range(r):
        assign_matrix[i, i * L:(i + 1) * L] = 1.0
    assign_rhs = np.ones(r)

    # capacity per location
    capacity_l = np.ceil(alpha * r).astype(int)
    capacity_matrix = np.zeros((L, r * L))
    for l in range(L):
        for i in range(r):
            capacity_matrix[l, i * L + l] = 1.0
    capacity_limits = capacity_l.astype(float)

    bounds = [(0, 1)] * (r * L)

    res = linprog(c, A_ub=capacity_matrix, b_ub=capacity_limits,
                  A_eq=assign_matrix, b_eq=assign_rhs,
                  bounds=bounds, method="highs")

    if not res.success:
        # fallback random si la LP foire
        print("LP failed:", res.message)
        x = np.zeros((r, L))
        for i in range(r):
            x[i, np.random.randint(L)] = 1.0
        return x

    return res.x.reshape(r, L)


def round_assignment(x_frac):
    # round to hard assignment
    r, L = x_frac.shape
    x_int = np.zeros_like(x_frac)
    for i in range(r):
        x_int[i, np.argmax(x_frac[i])] = 1.0
    return x_int


def run_one_seed(seed, verbose=False):
    np.random.seed(seed)

    true_model = TrueModel(verbose=verbose)
    predictor = LinearPredictor(random_state=seed)

    all_types, all_l, all_o = [], [], []

    emp_rate_learned = np.zeros(n_rounds)
    emp_rate_true = np.zeros(n_rounds)
    emp_rate_random = np.zeros(n_rounds)

    for t in range(n_rounds):
        n_t = np.random.randint(batch_min, batch_max + 1)
        types = generate_refugees(n_t)
        V_true = true_model.get_V(types)

        # learned path
        V_hat = predictor.predict_V(types)
        x_frac = solve_lp(V_hat, alpha)
        assignment = round_assignment(x_frac)

        outcomes = np.zeros(n_t)
        assigned_locs = np.zeros(n_t, dtype=int)
        for i in range(n_t):
            l = np.argmax(assignment[i])
            assigned_locs[i] = l
            outcomes[i] = np.random.binomial(1, V_true[i, l])

        # update history + retrain
        all_types.append(types)
        all_l.append(assigned_locs)
        all_o.append(outcomes)
        history_types = np.concatenate(all_types)
        history_l = np.concatenate(all_l)
        history_o = np.concatenate(all_o)
        predictor.train(history_types, history_l, history_o)

        # oracle (knows V_true)
        x_true_frac = solve_lp(V_true, alpha)
        assign_true = round_assignment(x_true_frac)
        outcomes_true = np.zeros(n_t)
        for i in range(n_t):
            l = np.argmax(assign_true[i])
            outcomes_true[i] = np.random.binomial(1, V_true[i, l])

        # random baseline
        outcomes_rand = np.zeros(n_t)
        for i in range(n_t):
            l = np.random.randint(n_locations)
            outcomes_rand[i] = np.random.binomial(1, V_true[i, l])

        emp_rate_learned[t] = outcomes.mean()
        emp_rate_true[t] = outcomes_true.mean()
        emp_rate_random[t] = outcomes_rand.mean()

    return emp_rate_learned, emp_rate_true, emp_rate_random


def run_simulation():
    print(f"Running simulation: {n_seeds} seeds x {n_rounds} rounds, "
          f"{n_locations} locations (affiliates {locations}), "
          f"{n_types} refugee types")
    print("-" * 80)

    TrueModel(verbose=True)
    print()

    learned_all = np.zeros((n_seeds, n_rounds))
    true_all = np.zeros((n_seeds, n_rounds))
    random_all = np.zeros((n_seeds, n_rounds))

    for s in range(n_seeds):
        learned_all[s], true_all[s], random_all[s] = run_one_seed(seed=s)
        print(f"  Seed {s+1:3d}/{n_seeds}  |  "
              f"Learned: {learned_all[s].mean():.3f}  |  "
              f"True oracle: {true_all[s].mean():.3f}  |  "
              f"Random: {random_all[s].mean():.3f}")

    rounds_axis = np.arange(1, n_rounds + 1)
    cum_learned = np.cumsum(learned_all, axis=1) / rounds_axis
    cum_true    = np.cumsum(true_all, axis=1)    / rounds_axis
    cum_random  = np.cumsum(random_all, axis=1)  / rounds_axis

    mean_learned, std_learned = cum_learned.mean(axis=0), cum_learned.std(axis=0)
    mean_true,    std_true    = cum_true.mean(axis=0),    cum_true.std(axis=0)
    mean_random,  std_random  = cum_random.mean(axis=0),  cum_random.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(rounds_axis, mean_true, "s--", label="True oracle",
            linewidth=2, color="tab:orange")
    ax.fill_between(rounds_axis, mean_true - std_true, mean_true + std_true,
                    alpha=0.2, color="tab:orange")

    ax.plot(rounds_axis, mean_learned, "o-", label="Learned predictor",
            linewidth=2, color="tab:blue")
    ax.fill_between(rounds_axis, mean_learned - std_learned, mean_learned + std_learned,
                    alpha=0.2, color="tab:blue")

    ax.plot(rounds_axis, mean_random, "x:", label="Random allocation",
            linewidth=1.5, alpha=0.8, color="tab:gray")
    ax.fill_between(rounds_axis, mean_random - std_random, mean_random + std_random,
                    alpha=0.15, color="tab:gray")

    ax.set_xlabel("Round t")
    ax.set_xticks(rounds_axis[::5])
    ax.set_ylabel("Cumulative employment rate")
    ax.set_title(f"Integration outcomes "
                 f"(mean $\\pm$ 1 std over {n_seeds} seeds)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150)
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    print("\nPlot saved to simulation_results.png")

    print("\n" + "=" * 80)
    print(f"  Avg employment (learned):      {cum_learned[:, -1].mean():.3f}  "
          f"(std {cum_learned[:, -1].std():.3f})")
    print(f"  Avg employment (true oracle):  {cum_true[:, -1].mean():.3f}  "
          f"(std {cum_true[:, -1].std():.3f})")
    print(f"  Avg employment (random):       {cum_random[:, -1].mean():.3f}  "
          f"(std {cum_random[:, -1].std():.3f})")
    print(f"  Gap learned vs oracle:         "
          f"{cum_true[:, -1].mean() - cum_learned[:, -1].mean():.3f}")
    print(f"  Gap learned vs random:         "
          f"{cum_learned[:, -1].mean() - cum_random[:, -1].mean():.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_simulation()
