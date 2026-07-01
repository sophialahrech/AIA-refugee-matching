# Rate-limited matching LP (Defence 1). We bound how much the type-conditional
# allocation p(k,l) can move between rounds. Two versions: a hard constraint
# |p_t - p_{t-1}| <= eps, and a soft L1 penalty rho on the same drift.

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def _load_sk():
    # the karma-tier model is only needed for the demo driver below
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "modelling_with_karma"))
    import simu_karma_tiers as sk
    return sk


def solve_rate_limited_lp(scores, types, alpha, p_prev, eps, n_types):
    r, T = scores.shape
    c = (-scores + np.random.uniform(0, 1e-8, size=scores.shape)).flatten()

    A_eq = np.zeros((r, r * T))
    for i in range(r):
        A_eq[i, i * T:(i + 1) * T] = 1.0
    b_eq = np.ones(r)

    cap = np.ceil(alpha * r).astype(int).astype(float)
    A_ub_cap = np.zeros((T, r * T))
    for t in range(T):
        for i in range(r):
            A_ub_cap[t, i * T + t] = 1.0

    # rate-limit inequalities: N_k*(p_prev - eps) <= sum_i x[i,t] <= N_k*(p_prev + eps)
    extra_rows, extra_rhs = [], []
    if eps < 1.0:
        for k in range(n_types):
            idx_k = np.flatnonzero(types == k)
            N_k = len(idx_k)
            if N_k == 0:
                continue
            for t in range(T):
                row = np.zeros(r * T)
                for i in idx_k:
                    row[i * T + t] = 1.0
                extra_rows.append(row)
                extra_rhs.append(N_k * (p_prev[k, t] + eps))
                extra_rows.append(-row)
                extra_rhs.append(-N_k * max(p_prev[k, t] - eps, 0.0))

    if extra_rows:
        A_ub = np.vstack([A_ub_cap, np.vstack(extra_rows)])
        b_ub = np.concatenate([cap, np.asarray(extra_rhs)])
    else:
        A_ub = A_ub_cap
        b_ub = cap

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, 1)] * (r * T), method="highs")
    if res.success:
        return res.x.reshape(r, T), True

    # infeasible rate-limit: fall back to capacity-only
    res2 = linprog(c, A_ub=A_ub_cap, b_ub=cap, A_eq=A_eq, b_eq=b_eq,
                   bounds=[(0, 1)] * (r * T), method="highs")
    warnings.warn(f"rate-limited LP infeasible at eps={eps}, "
                  "fell back to capacity-only.")
    return res2.x.reshape(r, T), False



def solve_soft_rate_limited_lp(scores, types, alpha, p_prev, rho, n_types):
    # Maximise sum(scores * x) - rho * sum_{k,l} |p(k,l) - p_prev(k,l)|,
    # the L1 penalty linearised with auxiliary variables y >= |Delta p|.
    r, T = scores.shape
    n_x = r * T
    n_y = n_types * T
    n_var = n_x + n_y

    # objective: -scores . x + rho * sum y
    c = np.zeros(n_var)
    c[:n_x] = (-scores + np.random.uniform(0, 1e-8, size=scores.shape)).flatten()
    c[n_x:] = rho

    # equality: each refugee assigned to one tier (sum_l x[i,l] = 1)
    A_eq = np.zeros((r, n_var))
    for i in range(r):
        A_eq[i, i * T:(i + 1) * T] = 1.0
    b_eq = np.ones(r)

   
    cap = np.ceil(alpha * r).astype(int).astype(float)
    rows_cap = np.zeros((T, n_var))
    for t in range(T):
        for i in range(r):
            rows_cap[t, i * T + t] = 1.0

    pen_rows, pen_rhs = [], []
    for k in range(n_types):
        idx_k = np.flatnonzero(types == k)
        N_k = len(idx_k)
        if N_k == 0:
            continue
        for l in range(T):
           
            row_u = np.zeros(n_var)
            for i in idx_k:
                row_u[i * T + l] = 1.0 / N_k
            row_u[n_x + k * T + l] = -1.0
            pen_rows.append(row_u)
            pen_rhs.append(p_prev[k, l])
            
            row_l = np.zeros(n_var)
            for i in idx_k:
                row_l[i * T + l] = -1.0 / N_k
            row_l[n_x + k * T + l] = -1.0
            pen_rows.append(row_l)
            pen_rhs.append(-p_prev[k, l])

    A_ub = np.vstack([rows_cap] + ([np.vstack(pen_rows)] if pen_rows else []))
    b_ub = np.concatenate([cap] + ([np.asarray(pen_rhs)] if pen_rhs else []))

    bounds = [(0, 1)] * n_x + [(0, None)] * n_y
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return res.x[:n_x].reshape(r, T), res.success



def compute_p_conditional(assignment, types, n_types, n_locs):

    p = np.zeros((n_types, n_locs))
    for k in range(n_types):
        idx = np.flatnonzero(types == k)
        if len(idx) == 0:
            continue
        p[k] = assignment[idx].mean(axis=0)
    return p


def run_one_seed_rate_limited(seed, model, attack_levels_vec, knob,
                              mode="soft", p_prev_init=None):
    """No-karma simulation with rate-limited matching.

    mode = "hard": knob is eps (rate budget), uses hard constraint with fallback.
    mode = "soft": knob is rho (penalty weight), uses soft penalty (always feasible).
    """
    sk = _load_sk()
    np.random.seed(seed)
    n_types = sk.N_TYPES
    n_locs = sk.N_TIERS
    arrivals = model["quarterly_arrivals"]
    n_rounds = len(arrivals)

    predictor = sk.TierPredictor(random_state=seed)

    if p_prev_init is None:
        p_prev = np.tile(model["alpha"][None, :], (n_types, 1))
    else:
        p_prev = p_prev_init.copy()

    util_hist = np.zeros((n_rounds, n_locs))
    emp_hist  = np.zeros(n_rounds)
    p_hist    = np.zeros((n_rounds, n_types, n_locs))
    drift_lp_hist    = np.zeros(n_rounds)
    drift_round_hist = np.zeros(n_rounds)
    n_infeasible = 0

    types_buf, tier_buf, out_buf = [], [], []

    for t in range(n_rounds):
        n_t = int(arrivals[t])
        types = np.random.choice(n_types, size=n_t, p=sk.TYPE_PROBS)
        V_true = model["theta"][types]
        lambdas = sk.location_attacks(types, attack_levels_vec,
                                      model["cost_mask"])
        V_hat = predictor.predict(types)

       
        if t == 0:
            x_frac, ok = solve_rate_limited_lp(V_hat, types, model["alpha"],
                                               p_prev, 1.0, n_types)
        elif mode == "hard":
            x_frac, ok = solve_rate_limited_lp(V_hat, types, model["alpha"],
                                               p_prev, knob, n_types)
        elif mode == "soft":
            x_frac, ok = solve_soft_rate_limited_lp(V_hat, types, model["alpha"],
                                                    p_prev, knob, n_types)
        else:
            raise ValueError(f"unknown mode {mode}")
        if not ok:
            n_infeasible += 1

        assignment = sk.round_assignment(x_frac)
        assigned = assignment.argmax(axis=1)

        outcomes = np.zeros(n_t)
        for i in range(n_t):
            l = assigned[i]
            p = (1 - lambdas[i, l]) * V_true[i, l]
            outcomes[i] = np.random.binomial(1, p)

        
        types_buf.append(types); tier_buf.append(assigned); out_buf.append(outcomes)
        predictor.train(np.concatenate(types_buf),
                        np.concatenate(tier_buf),
                        np.concatenate(out_buf))


        p_frac  = compute_p_conditional(x_frac,    types, n_types, n_locs)
        p_round = compute_p_conditional(assignment, types, n_types, n_locs)

        appeared = np.array([(types == k).sum() > 0 for k in range(n_types)])
        if t > 0 and appeared.any():
            drift_lp_hist[t]    = np.abs(p_frac[appeared]  - p_prev[appeared]).max()
            drift_round_hist[t] = np.abs(p_round[appeared] - p_prev[appeared]).max()

        for k in range(n_types):
            if appeared[k]:
                p_prev[k] = p_frac[k]

        util_hist[t] = sk.location_utility(types, assigned, outcomes,
                                           model["cost_mask"]) / n_t
        emp_hist[t]  = outcomes.mean()
        p_hist[t]    = p_prev

    return (emp_hist, util_hist, p_hist,
            drift_lp_hist, drift_round_hist, n_infeasible)



RHO_GRID = [0.0, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0]
N_SEEDS  = 8


def sweep_knob(model, attack, knob_grid, mode="soft", n_seeds=N_SEEDS):
    rows = []
    n_rounds = len(model["quarterly_arrivals"])
    for knob in knob_grid:
        emps, drifts_lp, drifts_round, infeas = [], [], [], []
        for s in range(n_seeds):
            e, _u, _p, d_lp, d_round, n_inf = run_one_seed_rate_limited(
                s, model, attack, knob, mode=mode)
            half = len(e) // 2
            emps.append(e[half:].mean())
            drifts_lp.append(d_lp[half:].mean())
            drifts_round.append(d_round[half:].mean())
            infeas.append(n_inf)
        rows.append(dict(
            knob=knob,
            emp_mean=float(np.mean(emps)),
            emp_std=float(np.std(emps)),
            drift_lp=float(np.mean(drifts_lp)),
            drift_round=float(np.mean(drifts_round)),
            n_infeasible=float(np.mean(infeas)),
        ))
        knob_label = "rho" if mode == "soft" else "eps"
        print(f"  {knob_label}={knob:6.2f} | emp={rows[-1]['emp_mean']:.4f} "
              f"+/- {rows[-1]['emp_std']:.4f} | "
              f"drift_LP={rows[-1]['drift_lp']:.4f} | "
              f"drift_round={rows[-1]['drift_round']:.4f} | "
              f"infeas={rows[-1]['n_infeasible']:.1f}/{n_rounds}")
    return rows


def plot_tradeoff(results_by_regime, out_path, knob_label="rho"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = {"truthful (lambda=0)": "#2ca02c",
              "Nash (lambda=1,1,1)": "#d62728"}
    for label, rows in results_by_regime.items():
        knob     = [r["knob"]        for r in rows]
        emp      = [r["emp_mean"]    for r in rows]
        emp_std  = [r["emp_std"]     for r in rows]
        d_lp     = [r["drift_lp"]    for r in rows]
        d_round  = [r["drift_round"] for r in rows]
        c = colors.get(label, "k")

        axes[0].errorbar(d_round, emp, yerr=emp_std, marker="o", color=c,
                         label=label, capsize=3, linewidth=2)
       
        axes[1].plot(knob, d_lp, marker="s", color=c,
                     label=f"{label} (LP)", linewidth=2)
        axes[1].plot(knob, d_round, marker="^", color=c,
                     label=f"{label} (rounded)", linewidth=2, linestyle="--")
        # efficiency vs knob
        axes[2].errorbar(knob, emp, yerr=emp_std, marker="o", color=c,
                         label=label, capsize=3, linewidth=2)

    axes[0].set_xlabel(r"realised $\|\Delta p\|_\infty$ per round (post-rounding)")
    axes[0].set_ylabel("mean employment rate ")
    axes[0].set_title("efficiency vs realised smoothness")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    knob_ax = f"penalty weight $\\{knob_label}$" if knob_label == "rho" \
              else r"rate-limit budget $\varepsilon$"
    for ax in (axes[1], axes[2]):
        ax.set_xlabel(knob_ax)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].set_ylabel(r"$\|\Delta p\|_\infty$ per round")
    axes[1].set_title("Drift vs penalty weight")
    axes[2].set_ylabel("mean employment rate (2nd half)")
    axes[2].set_title("Efficiency vs penalty weight")

    if knob_label == "rho":
        # rho is on a log-ish scale, but includes 0; use symlog
        for ax in (axes[1], axes[2]):
            ax.set_xscale("symlog", linthresh=0.1)

    plt.suptitle("Rate-limited matching: stability / efficiency trade-off",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Trade-off plot saved to {out_path}")



if __name__ == "__main__":
    print("=" * 78)
    print("Rate-limited matching (Task 1) -- SOFT formulation")
    print(f"  rho grid = {RHO_GRID}")
    print(f"  n_seeds  = {N_SEEDS}")
    print("=" * 78)

    sk = _load_sk()
    ATTACK_REGIMES = {
        "truthful (lambda=0)":  np.zeros(sk.N_TIERS),
        "Nash (lambda=1,1,1)":  np.ones(sk.N_TIERS),
    }
    print(f"  regimes  = {list(ATTACK_REGIMES.keys())}")

    model = sk.build_tier_model()

    results_by_regime = {}
    for label, attack in ATTACK_REGIMES.items():
        print(f"\n[{label}]")
        results_by_regime[label] = sweep_knob(model, attack, RHO_GRID,
                                              mode="soft")

    here = os.path.dirname(__file__)
    plot_tradeoff(results_by_regime,
                  os.path.join(here, "rate_limited_soft_tradeoff.png"),
                  knob_label="rho")

    print("\n" + "=" * 78)
    print("Summary (SOFT formulation)")
    print("=" * 78)
    for label, rows in results_by_regime.items():
        emp_unconstrained = next(r["emp_mean"] for r in rows if r["knob"] == 0.0)
        print(f"\n  {label}:")
        print(f"    rho=0  (no penalty):  emp = {emp_unconstrained:.4f}")
        for r in rows:
            if r["knob"] == 0.0:
                continue
            delta = r["emp_mean"] - emp_unconstrained
            pct = 100 * delta / max(emp_unconstrained, 1e-9)
            print(f"    rho={r['knob']:6.2f}:  emp = {r['emp_mean']:.4f}  "
                  f"({pct:+.1f}% vs unconstrained)  "
                  f"drift_LP = {r['drift_lp']:.3f}  "
                  f"drift_round = {r['drift_round']:.3f}")
