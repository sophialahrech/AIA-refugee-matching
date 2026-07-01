# simu1 + adversarial attacks 
# o_rl(lambda) ~ Bernoulli((1 - lambda_l(u)) * V_rl)

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

type_probs = np.array(
    [6048, 6272, 2816, 5632, 12128, 19264, 5344, 19424], dtype=float
)
type_probs = type_probs / type_probs.sum()

MALE_MASK    = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
ENGLISH_MASK = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool)

n_rounds = 50
batch_min = 20
batch_max = 100
alpha = np.array([0.6, 0.4])
n_seeds = 50
attack_levels = np.array([0.9, 0.9])


def load_theta(locations):
    df = pd.read_csv(THETA_CSV)
    theta = np.zeros((n_types, len(locations)))
    for l, v in enumerate(locations):
        sub = df[df["V"] == v].sort_values("U")
        if len(sub) != n_types:
            raise ValueError(f"Affiliate V={v} does not have {n_types} U rows.")
        theta[:, l] = sub["theta"].values
    return theta

USE_ARBITRARY_MASKS = False


def _compute_cost_mask(theta):
    L = theta.shape[1]
    mask = np.zeros_like(theta, dtype=bool)
    for l in range(L):
        mask[:, l] = theta[:, l] < np.median(theta[:, l])
    return mask


_THETA_GROUND = load_theta(locations)
if USE_ARBITRARY_MASKS:
    LOCATION_ATTACK_MASK = np.stack([MALE_MASK, ENGLISH_MASK], axis=1)
else:
    LOCATION_ATTACK_MASK = _compute_cost_mask(_THETA_GROUND)

# utility parameters
# U_l = +reward * o_i  -  cost * 1{i costly for l}
EMPLOYMENT_REWARD = 1.0
INTEGRATION_COST = 1.0

# baseline no-attack 
_BASELINE_CACHE = {}

def compute_no_attack_baseline(n_seeds_baseline=None, verbose=True):
    import simu1
    if n_seeds_baseline is None:
        n_seeds_baseline = n_seeds
    if n_seeds_baseline in _BASELINE_CACHE:
        return _BASELINE_CACHE[n_seeds_baseline]
    rates = np.zeros(n_seeds_baseline)
    for s in range(n_seeds_baseline):
        emp_learned, _, _ = simu1.run_one_seed(seed=s, verbose=False)
        rates[s] = emp_learned.mean()
    val = float(rates.mean())
    if verbose:
        print(f"  [baseline] simu1 learned predictor over "
              f"{n_seeds_baseline} seeds: {val:.3f} (std {rates.std():.3f})")
    _BASELINE_CACHE[n_seeds_baseline] = val
    return val


def generate_refugees(r):
    return np.random.choice(n_types, size=r, p=type_probs)


class TrueModel:
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


def location_attacks(types, attack_levels_vec=None):
    # renvoie la matrice lambda (r, L)
    if attack_levels_vec is None:
        attack_levels_vec = attack_levels
    r = len(types)
    lambdas = np.zeros((r, n_locations))
    for l in range(n_locations):
        attacked_types = LOCATION_ATTACK_MASK[:, l]
        lambdas[:, l] = np.where(attacked_types[types],
                                 attack_levels_vec[l], 0.0)
    return lambdas


def location_utility(types, assigned_locs, outcomes):
    # U_l = sum sur les refugees assignés à l de (reward*o_i - cost*1{costly})
    util = np.zeros(n_locations)
    r = len(types)
    for i in range(r):
        l = assigned_locs[i]
        reward = EMPLOYMENT_REWARD * outcomes[i]
        costly = LOCATION_ATTACK_MASK[types[i], l]
        cost = INTEGRATION_COST if costly else 0.0
        util[l] += reward - cost
    return util


def solve_lp(V_hat, alpha):
    r, L = V_hat.shape
    # tiny noise to break ties in the LP objective. HiGHS is deterministic
    # and picks a lexicographic vertex in degenerate cases (systematic L0
    # bias). Perturbing V by epsilon << any meaningful diff breaks ties
    # without changing the real optimum. Uses np.random global state
    # (seeded in run_one_seed) for reproducibility.
    V_hat = V_hat + np.random.uniform(0, 1e-8, size=V_hat.shape)
    c = -V_hat.flatten()

    assign_matrix = np.zeros((r, r * L))
    for i in range(r):
        assign_matrix[i, i * L:(i + 1) * L] = 1.0
    assign_rhs = np.ones(r)

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
        print("LP failed:", res.message)
        x = np.zeros((r, L))
        for i in range(r):
            x[i, np.random.randint(L)] = 1.0
        return x

    return res.x.reshape(r, L)


def round_assignment(x_frac):
    # random tie-break when several columns have the same fractional max.
    # np.argmax returns the lowest index -> systematic L0 bias, especially
    # at (lambda_0, lambda_1) cells where V_strat collapses to 0 for types
    # costly to both locations (diagonal in strat-aware case).
    r, L = x_frac.shape
    x_int = np.zeros_like(x_frac)
    for i in range(r):
        row = x_frac[i]
        max_val = row.max()
        tied = np.flatnonzero(row >= max_val - 1e-9)
        choice = tied[0] if len(tied) == 1 else np.random.choice(tied)
        x_int[i, choice] = 1.0
    return x_int


def allocate_and_evaluate(V_for_lp, V_true, lambdas=None, sample=True):
    # LP avec V_for_lp, évalue sur V_true (attaques optionnelles)
    x_frac = solve_lp(V_for_lp, alpha)
    assignment = round_assignment(x_frac)
    r = V_true.shape[0]

    outcomes = np.zeros(r)
    for i in range(r):
        l = np.argmax(assignment[i])
        p = V_true[i, l]
        if lambdas is not None:
            p = (1 - lambdas[i, l]) * p
        if sample:
            outcomes[i] = np.random.binomial(1, p)
        else:
            outcomes[i] = p
    return outcomes.mean()


def run_one_seed(seed, verbose=False, attack_levels_vec=None,
                 return_refugee_data=False):
    np.random.seed(seed)

    if attack_levels_vec is None:
        attack_levels_vec = attack_levels

    true_model = TrueModel(verbose=verbose)
    predictor = LinearPredictor(random_state=seed)

    all_types, all_l, all_o = [], [], []

    emp_rate_learned = np.zeros(n_rounds)
    emp_rate_true = np.zeros(n_rounds)
    emp_rate_strategy_aware = np.zeros(n_rounds)
    emp_rate_random = np.zeros(n_rounds)
    utility_learned = np.zeros((n_rounds, n_locations))
    utility_oracle = np.zeros((n_rounds, n_locations))
    utility_strategy_aware = np.zeros((n_rounds, n_locations))

    # strat-aware oracle: il n'a pas d'info au round 0, mais des que t>=1
    # il a observe les outcomes attaques et a deduit le vecteur de strategie
    # des locations. Dans un jeu statique (attack_levels_vec fixe) ca revient
    # a connaitre le attack_levels_vec a partir de t=1.
    strat_known = False

    for t in range(n_rounds):
        n_t = np.random.randint(batch_min, batch_max + 1)
        types = generate_refugees(n_t)
        V_true = true_model.get_V(types)
        lambdas = location_attacks(types, attack_levels_vec)

        # learned path (avec attaques)
        V_hat = predictor.predict_V(types)
        x_frac = solve_lp(V_hat, alpha)
        assignment = round_assignment(x_frac)

        outcomes_attacked = np.zeros(n_t)
        assigned_locs = np.zeros(n_t, dtype=int)

        for i in range(n_t):
            l = np.argmax(assignment[i])
            assigned_locs[i] = l
            attacked_p = (1 - lambdas[i, l]) * V_true[i, l]
            outcomes_attacked[i] = np.random.binomial(1, attacked_p)

        # retrain sur outcomes observés (attaqués)
        all_types.append(types)
        all_l.append(assigned_locs)
        all_o.append(outcomes_attacked)
        history_types = np.concatenate(all_types)
        history_l = np.concatenate(all_l)
        history_o = np.concatenate(all_o)
        predictor.train(history_types, history_l, history_o)

        # oracles (tous mesurés avec les mêmes attaques)
        emp_true_val = allocate_and_evaluate(V_true, V_true, lambdas=lambdas,
                                             sample=True)

        # strategy-aware oracle: au round 0 il n'a rien observe -> V_true.
        # A partir de t>=1 il a deduit la strategie des locations (vecteur
        # attack_levels_vec + mask) en observant les outcomes au round t-1.
        # Il peut donc recomputer les lambdas pour les types CURRENT et
        # reallouer (les types attaques dans une location partent ailleurs).
        if not strat_known:
            V_strat = V_true
        else:
            V_strat = (1 - lambdas) * V_true
        x_frac_strat = solve_lp(V_strat, alpha)
        assignment_strat = round_assignment(x_frac_strat)
        outcomes_strat = np.zeros(n_t)
        assigned_locs_strat = np.zeros(n_t, dtype=int)
        for i in range(n_t):
            l = np.argmax(assignment_strat[i])
            assigned_locs_strat[i] = l
            attacked_p = (1 - lambdas[i, l]) * V_true[i, l]
            outcomes_strat[i] = np.random.binomial(1, attacked_p)
        emp_strat = outcomes_strat.mean()

        outcomes_rand = np.zeros(n_t)
        for i in range(n_t):
            l = np.random.randint(n_locations)
            attacked_p = (1 - lambdas[i, l]) * V_true[i, l]
            outcomes_rand[i] = np.random.binomial(1, attacked_p)

        # oracle avec V_true pour la LP, mais attaques toujours actives
        x_frac_oracle = solve_lp(V_true, alpha)
        assignment_oracle = round_assignment(x_frac_oracle)
        outcomes_oracle = np.zeros(n_t)
        assigned_locs_oracle = np.zeros(n_t, dtype=int)
        for i in range(n_t):
            l = np.argmax(assignment_oracle[i])
            assigned_locs_oracle[i] = l
            attacked_p = (1 - lambdas[i, l]) * V_true[i, l]
            outcomes_oracle[i] = np.random.binomial(1, attacked_p)

        emp_rate_learned[t] = outcomes_attacked.mean()
        emp_rate_true[t] = emp_true_val
        emp_rate_strategy_aware[t] = emp_strat
        emp_rate_random[t] = outcomes_rand.mean()

        utility_learned[t] = location_utility(types, assigned_locs,
                                              outcomes_attacked) / n_t
        utility_oracle[t] = location_utility(types, assigned_locs_oracle,
                                             outcomes_oracle) / n_t
        utility_strategy_aware[t] = location_utility(types, assigned_locs_strat,
                                                     outcomes_strat) / n_t

        strat_known = True

    if return_refugee_data:
        types_all = np.concatenate(all_types)
        locs_all = np.concatenate(all_l)
        outcomes_all = np.concatenate(all_o)
        round_sizes = np.array([len(a) for a in all_types])
        return (emp_rate_learned, emp_rate_true,
                emp_rate_strategy_aware, emp_rate_random,
                utility_learned, utility_oracle, utility_strategy_aware,
                types_all, locs_all, outcomes_all, round_sizes)

    return (emp_rate_learned, emp_rate_true,
            emp_rate_strategy_aware, emp_rate_random,
            utility_learned, utility_oracle, utility_strategy_aware)


def run_simulation():
    print(f"Running simulation: {n_seeds} seeds x {n_rounds} rounds, "
          f"{n_locations} locations (affiliates {locations}), "
          f"{n_types} refugee types, attack_levels={attack_levels}")
    print("-" * 100)

    TrueModel(verbose=True)
    print()

    print("Computing no-attack baseline from simu1...")
    no_attack_baseline = compute_no_attack_baseline()
    print()

    learned_all = np.zeros((n_seeds, n_rounds))
    true_all = np.zeros((n_seeds, n_rounds))
    strat_all = np.zeros((n_seeds, n_rounds))
    random_all = np.zeros((n_seeds, n_rounds))
    util_all = np.zeros((n_seeds, n_rounds, n_locations))
    util_oracle_all = np.zeros((n_seeds, n_rounds, n_locations))
    util_strat_all = np.zeros((n_seeds, n_rounds, n_locations))

    for s in range(n_seeds):
        (learned_all[s], true_all[s],
         strat_all[s], random_all[s],
         util_all[s], util_oracle_all[s], util_strat_all[s]) = run_one_seed(seed=s)
        print(f"  Seed {s+1:3d}/{n_seeds}  |  "
              f"Learned: {learned_all[s].mean():.3f}  |  "
              f"U0: {util_all[s].mean(axis=0)[0]:+.3f}  |  "
              f"U1: {util_all[s].mean(axis=0)[1]:+.3f}")

    rounds_axis = np.arange(1, n_rounds + 1)

    def cum(arr):
        return np.cumsum(arr, axis=1) / rounds_axis

    cum_learned = cum(learned_all)
    cum_true = cum(true_all)
    cum_strat = cum(strat_all)
    cum_random = cum(random_all)

    def mean_std(c):
        return c.mean(axis=0), c.std(axis=0)

    m_learned, s_learned = mean_std(cum_learned)
    m_true, s_true = mean_std(cum_true)
    m_strat, s_strat = mean_std(cum_strat)
    m_random, s_random = mean_std(cum_random)

    fig, ax = plt.subplots(figsize=(10, 6))

    def plot_band(mean, std, label, marker, color, alpha_band=0.18, lw=2):
        ax.plot(rounds_axis, mean, marker, label=label, linewidth=lw, color=color)
        ax.fill_between(rounds_axis, mean - std, mean + std,
                        alpha=alpha_band, color=color)

    plot_band(m_strat, s_strat, "Strategy-aware oracle", "^-", "tab:cyan")
    plot_band(m_true, s_true, "True oracle", "s--", "tab:orange")
    plot_band(m_learned, s_learned, "Learned predictor (attacked)", "o-", "tab:blue")
    plot_band(m_random, s_random, "Random allocation", "x:", "tab:gray",
              alpha_band=0.15, lw=1.5)

    ax.axhline(no_attack_baseline, color="tab:red", linestyle="--",
               linewidth=1.5, alpha=0.8,
               label=f"No-attack baseline (simu1): {no_attack_baseline:.3f}")

    ax.set_xlabel("Round t")
    ax.set_xticks(rounds_axis[::5])
    ax.set_ylabel("Cumulative employment rate (under attacks)")
    ax.set_title(f"Learned predictor vs oracles under attacks "
                 f"(affiliates {locations}, mean $\\pm$ 1 std over {n_seeds} seeds)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_attack_results.png", dpi=150)
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    print("\nPlot saved to simulation_attack_results.png")

    learned_end = cum_learned[:, -1]
    true_end = cum_true[:, -1]
    strat_end = cum_strat[:, -1]
    random_end = cum_random[:, -1]

    half = n_rounds // 2
    util_steady = util_all[:, half:, :].mean(axis=1)
    util_mean = util_steady.mean(axis=0)
    util_std = util_steady.std(axis=0)

    print("\n" + "=" * 80)
    print("  All metrics measured under attacks.")
    print("-" * 80)
    print(f"  Learned predictor (attacked):    {learned_end.mean():.3f}  "
          f"(std {learned_end.std():.3f})")
    print(f"  True oracle (Bernoulli):         {true_end.mean():.3f}  "
          f"(std {true_end.std():.3f})")
    print(f"  Strategy-aware oracle:           {strat_end.mean():.3f}  "
          f"(std {strat_end.std():.3f})")
    print(f"  Random allocation:               {random_end.mean():.3f}  "
          f"(std {random_end.std():.3f})")
    print("-" * 80)
    print("  Per-location utility (per refugee, steady state):")
    for l in range(n_locations):
        print(f"    Location {l} (lambda={attack_levels[l]:.2f}):  "
              f"U = {util_mean[l]:+.3f}  (std {util_std[l]:.3f})")
    print("-" * 80)
    print(f"  No-attack baseline (simu1):      {no_attack_baseline:.3f}")
    print(f"  Attack damage on learned:        "
          f"{no_attack_baseline - learned_end.mean():+.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_simulation()
