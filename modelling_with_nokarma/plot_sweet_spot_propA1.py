# Sweet-spot unilateral interaction-proofness test (Fig 7 analogue) under
# the generalised Proposition A.1 per-individual utility + discounted cumulative
# discount across all T simulation rounds.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11.5,
})

sys.path.insert(0, os.path.dirname(__file__))
from simu2_real import (build_2loc_model, location_attacks, round_assignment,
                        N_TYPES, TYPE_PROBS, LocPredictor, per_loc_utility,
                        C_APPENDIX)
from matching_rate_limited import (solve_rate_limited_lp,
                                   solve_soft_rate_limited_lp,
                                   compute_p_conditional)


LOC_A, LOC_B = 14, 4

BETA = 0.9

# Planner used for the rate-limit sweep.
#   "strat"    = strategy-aware (knows lambda_t, instantly reroutes).
#   "learned"  = naive logistic learner (cold start, slow).
#   "truthful" = pure observer with oracle access to V_true.
#                Knows lambda_{t-1} (informational, used for tracking only).
#                Always scores with V_true, never reacts to attacks.
PLANNER = "truthful"

EPS_GRID = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0]
N_SEEDS  = 20

CACHE = os.path.join(os.path.dirname(__file__),
                     f"sweet_spot_propA1_{PLANNER}_beta{int(BETA*100):02d}_"
                     f"{LOC_A}_{LOC_B}.npz")


def run_one_seed_rate_limited_propA1(seed, model, attack_levels_vec, knob,
                                     beta, c=C_APPENDIX,
                                     mode="hard", planner="learned"):
    """Per-round utility uses Proposition A.1 (depends on p_{k,l} and lambda).
    Returns util_hist of shape (n_rounds, n_loc), the SUM of per-individual
    Prop A.1 utility at each location, per round.
    """
    np.random.seed(seed)
    n_types = N_TYPES
    n_loc   = 2
    arrivals = model["quarterly_arrivals"]
    n_rounds = len(arrivals)
    cost_mask = model["cost_mask"]
    alpha     = model["alpha"]
    theta     = model["theta"]

    predictor = LocPredictor(n_loc, random_state=seed)
    p_prev    = np.tile(alpha[None, :], (n_types, 1))

    # Truthful planner: at round t, scores with (1 - lambda_{t-1}) * V_true.
    # At t=0 lambda_prev is zero (nothing observed yet).
    lambda_prev_kl = np.zeros((n_types, n_loc))

    emp_hist         = np.zeros(n_rounds)
    drift_round_hist = np.zeros(n_rounds)
    util_hist        = np.zeros((n_rounds, n_loc))
    n_infeasible     = 0
    types_buf, loc_buf, out_buf = [], [], []

    for t in range(n_rounds):
        n_t = int(arrivals[t])
        if n_t == 0:
            continue
        types  = np.random.choice(n_types, size=n_t, p=TYPE_PROBS)
        V_true = theta[types]
        lambdas = location_attacks(types, attack_levels_vec, cost_mask)

        if planner == "learned":
            scores = predictor.predict(types)
        elif planner == "strat":
            scores = (1.0 - lambdas) * V_true
        else:
            # Truthful planner with oracle access to V_true.
            # At round t, it knows lambda from round t-1 and uses it as a
            # proxy for the current attack: compensates by scoring with
            #     (1 - lambda_{t-1}) * V_true.
            # At t=0 lambda_prev is zero, so the planner is uninformed and
            # the attacker gets one "free" round before compensation kicks in.
            scores = (1.0 - lambda_prev_kl[types]) * V_true

        if t == 0:
            x_frac, ok = solve_rate_limited_lp(scores, types, alpha,
                                                p_prev, 1.0, n_types)
        elif mode == "hard":
            x_frac, ok = solve_rate_limited_lp(scores, types, alpha,
                                                p_prev, knob, n_types)
            if not ok:
                # The rate-limited LP was infeasible. Instead of letting
                # solve_rate_limited_lp fall back to the UNCONSTRAINED LP
                # (which would make the simulation pretend nothing is
                # constrained), we honor the rate-limit semantically by
                # sampling each refugee from the previous conditional
                # allocation p_prev[type, :]. This is the genuine "frozen
                # planner" behavior the user wants to demonstrate.
                x_frac = np.zeros((n_t, n_loc))
                for i in range(n_t):
                    k = types[i]
                    probs = p_prev[k].copy()
                    if probs.sum() <= 0:
                        probs = np.ones(n_loc) / n_loc
                    else:
                        probs = probs / probs.sum()
                    l = np.random.choice(n_loc, p=probs)
                    x_frac[i, l] = 1.0
        elif mode == "soft":
            x_frac, ok = solve_soft_rate_limited_lp(scores, types, alpha,
                                                     p_prev, knob, n_types)
        else:
            raise ValueError(f"Unknown mode={mode!r}")
        if not ok:
            n_infeasible += 1

        assignment = round_assignment(x_frac)
        assigned   = assignment.argmax(axis=1)

        outcomes = np.array([
            np.random.binomial(1, (1 - lambdas[i, assigned[i]]) * V_true[i, assigned[i]])
            for i in range(n_t)
        ])

        types_buf.append(types); loc_buf.append(assigned); out_buf.append(outcomes)
        if planner == "learned":
            predictor.train(np.concatenate(types_buf),
                            np.concatenate(loc_buf),
                            np.concatenate(out_buf))
        elif planner == "truthful":
    
            for k in range(n_types):
                mask_k = (types == k)
                if mask_k.any():
                    lambda_prev_kl[k] = lambdas[mask_k].mean(axis=0)

        p_frac  = compute_p_conditional(x_frac,    types, n_types, n_loc)
        p_round = compute_p_conditional(assignment, types, n_types, n_loc)
        appeared = np.array([(types == k).sum() > 0 for k in range(n_types)])
        if t > 0 and appeared.any():
            drift_round_hist[t] = np.abs(p_round[appeared] - p_prev[appeared]).max()
        for k in range(n_types):
            if appeared[k]:
                p_prev[k] = p_frac[k]

        emp_hist[t] = outcomes.mean()
        util_hist[t] = per_loc_utility(types, assigned, cost_mask,
                                        c=c, beta_indiv=beta,
                                        theta=theta,
                                        attack_levels_vec=attack_levels_vec)

    return emp_hist, drift_round_hist, n_infeasible, util_hist



def discounted_cumulative(util_hist, beta):
    """Given per-round per-loc utility, return the discounted cumulative
    discounted utility over the full horizon T (no normalisation):
        U_l = sum_{t=0}^{T-1} beta^t * util_hist[t, l]
    """
    T = util_hist.shape[0]
    weights = beta ** np.arange(T)
    return (util_hist * weights[:, None]).sum(axis=0)


ATTACK_REGIMES = {
    "truthful (lambda=0), learned":          (np.array([0.0, 0.0]), PLANNER),
    "L0 unilateral (lambda=1,0), learned":   (np.array([1.0, 0.0]), PLANNER),
    "L1 unilateral (lambda=0,1), learned":   (np.array([0.0, 1.0]), PLANNER),
    "Nash (lambda=1,1), learned":            (np.array([1.0, 1.0]), PLANNER),
}


def sweep_one_regime(model, attack, planner, eps_grid, n_seeds, beta):
    rows = []
    for eps in eps_grid:
        u0s, u1s = [], []
        for s in range(n_seeds):
            _, _, _, u_hist = run_one_seed_rate_limited_propA1(
                s, model, attack, eps, beta, mode="hard", planner=planner)
            U = discounted_cumulative(u_hist, beta)
            u0s.append(U[0])
            u1s.append(U[1])
        rows.append(dict(eps=eps,
                         u0_mean=float(np.mean(u0s)),
                         u0_std=float(np.std(u0s)),
                         u1_mean=float(np.mean(u1s)),
                         u1_std=float(np.std(u1s))))
        print(f"    eps={eps:7.3f}  "
              f"U=({rows[-1]['u0_mean']:+.2f}, {rows[-1]['u1_mean']:+.2f})  "
              f"std=({rows[-1]['u0_std']:.2f}, {rows[-1]['u1_std']:.2f})")
    return rows


def compute_or_load(use_cache=True):
    prefixes = ["truth_learned", "uni_L0_learned",
                "uni_L1_learned", "nash_learned"]
    fields   = ["u0", "u0_std", "u1", "u1_std"]
    expected = {"eps_grid", "n_seeds", "beta"} | {
        f"{fld}_{pre}" for pre in prefixes for fld in fields
    }
    if use_cache and os.path.exists(CACHE):
        d = np.load(CACHE)
        if (expected.issubset(d.files)
                and np.allclose(d["eps_grid"], EPS_GRID)
                and int(d["n_seeds"]) == N_SEEDS
                and float(d["beta"]) == BETA):
            print(f"  Loaded cache from {CACHE}")
            def unpack(prefix):
                return [dict(eps=e_,
                             u0_mean=u0, u0_std=u0s,
                             u1_mean=u1, u1_std=u1s)
                        for e_, u0, u0s, u1, u1s in zip(
                            d["eps_grid"],
                            d[f"u0_{prefix}"], d[f"u0_std_{prefix}"],
                            d[f"u1_{prefix}"], d[f"u1_std_{prefix}"])]
            return {p: unpack(p) for p in prefixes}
        print("  Cache stale or missing fields, recomputing.")

    model = build_2loc_model(LOC_A, LOC_B)
    print(f"  Built 2-loc model: locs={model['locs']}, "
          f"cap={model['cap'].tolist()}, alpha={model['alpha'].round(3).tolist()}")

    label_keys = {
        "truthful (lambda=0), learned":            "truth_learned",
        "L0 unilateral (lambda=1,0), learned":     "uni_L0_learned",
        "L1 unilateral (lambda=0,1), learned":     "uni_L1_learned",
        "Nash (lambda=1,1), learned":              "nash_learned",
    }
    results = {}
    for label, (attack, planner) in ATTACK_REGIMES.items():
        print(f"\n  Sweeping eps_grid -- {label}")
        results[label_keys[label]] = sweep_one_regime(
            model, attack, planner, EPS_GRID, N_SEEDS, BETA)

    def col(rs, key):
        return np.array([r[key] for r in rs])

    np.savez(CACHE,
             eps_grid=np.array(EPS_GRID), n_seeds=N_SEEDS, beta=BETA,
             **{f"u0_{k}":     col(v, "u0_mean")  for k, v in results.items()},
             **{f"u0_std_{k}": col(v, "u0_std")   for k, v in results.items()},
             **{f"u1_{k}":     col(v, "u1_mean")  for k, v in results.items()},
             **{f"u1_std_{k}": col(v, "u1_std")   for k, v in results.items()})
    print(f"  Cached -> {CACHE}")
    return results



def plot_unilateral_incentive(results):
    eps_grid = np.array([r["eps"] for r in results["truth_learned"]])

    def col_u(key, loc):
        m = np.array([r[f"u{loc}_mean"] for r in results[key]])
        s = np.array([r[f"u{loc}_std"]  for r in results[key]])
        return m, s

    u0_T,  u0_T_s  = col_u("truth_learned", 0)
    u1_T,  u1_T_s  = col_u("truth_learned", 1)
    u0_uni, u0_uni_s = col_u("uni_L0_learned", 0)
    u1_uni, u1_uni_s = col_u("uni_L1_learned", 1)
    u0_nL, u0_nL_s = col_u("nash_learned", 0)
    u1_nL, u1_nL_s = col_u("nash_learned", 1)

    prem_uni_L0 = u0_uni - u0_T
    prem_uni_L1 = u1_uni - u1_T

    print(f"\n  Unilateral attack premium for L0 across eps:")
    for e_, p in zip(eps_grid, prem_uni_L0):
        print(f"    eps={e_:7.3f}  U_L0(1,0) - U_L0(0,0) = {p:+.3f}")
    print(f"\n  Unilateral attack premium for L1 across eps:")
    for e_, p in zip(eps_grid, prem_uni_L1):
        print(f"    eps={e_:7.3f}  U_L1(0,1) - U_L1(0,0) = {p:+.3f}")

    XTICKS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    XLABS  = ["1\n(no bound)", "0.5", "0.2", "0.1", "0.05", "0.02",
              "0.01", "0.005", "0.001\n(frozen)"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.5, 9.5),
                                    sharex=True,
                                    gridspec_kw={"hspace": 0.20})

    def _draw_panel(ax, u_T, u_T_s, u_uni, u_uni_s, u_n, u_n_s,
                    label_loc, who, show_legend):
        ax.errorbar(eps_grid, u_T, yerr=u_T_s,
                    marker="o", color="#2ca02c", lw=2.6, markersize=7,
                    capsize=4, alpha=0.95,
                    label=r"truthful $\lambda=(0,0)$")
        ax.errorbar(eps_grid, u_uni, yerr=u_uni_s,
                    marker="s", color="#d62728", lw=2.6, markersize=7,
                    capsize=4, alpha=0.95, linestyle="--",
                    label=r"unilateral attack")
        ax.errorbar(eps_grid, u_n, yerr=u_n_s,
                    marker="D", color="gray", lw=1.6, markersize=5,
                    capsize=3, alpha=0.65, linestyle=":",
                    label=r"joint Nash $\lambda=(1,1)$ (ref.)")
        ax.fill_between(eps_grid, u_T, u_uni, where=(u_uni >= u_T),
                        color="#d62728", alpha=0.18)
        ax.fill_between(eps_grid, u_T, u_uni, where=(u_uni < u_T),
                        color="#2ca02c", alpha=0.15)

        ax.set_xscale("log")
        ax.set_xticks(XTICKS); ax.set_xticklabels(XLABS)
        ax.invert_xaxis()
        ax.set_xlim(1.4, 0.0007)
        ax.minorticks_off()
        ax.set_ylabel(f"$U_{{{label_loc}}}$")
        ax.grid(alpha=0.3)
        if show_legend:
            ax.legend(loc="lower left", framealpha=0.95, ncol=1)
        ax.set_title(who, loc="left")

    _draw_panel(ax0, u0_T, u0_T_s, u0_uni, u0_uni_s, u0_nL, u0_nL_s,
                "\\ell_0",
                f"(a)  $\\ell_0$ = Loc {LOC_A}  attacks "
                "($\\lambda = (1, 0)$)",
                show_legend=True)
    _draw_panel(ax1, u1_T, u1_T_s, u1_uni, u1_uni_s, u1_nL, u1_nL_s,
                "\\ell_1",
                f"(b)  $\\ell_1$ = Loc {LOC_B}  attacks "
                "($\\lambda = (0, 1)$)",
                show_legend=False)

    ax1.set_xlabel(r"Update bound  $\varepsilon$")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
                       f"sweet_spot_unilateral_propA1_{PLANNER}_"
                       f"beta{int(BETA*100):02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # Caption
    joint_ok = (prem_uni_L0 <= 0) & (prem_uni_L1 <= 0)
    if joint_ok.any():
        eps_star = float(eps_grid[joint_ok][0])
        sweet_msg = (f"The largest $\\varepsilon$ at which both unilateral "
                     f"premia are non-positive (interaction-proof in the "
                     f"strict sense) is $\\varepsilon^\\star \\approx "
                     f"{eps_star:g}$.")
    else:
        sweet_msg = ("No $\\varepsilon$ on the grid yields both unilateral "
                     "premia non-positive simultaneously.")
    _planner_label = {"strat": "strategy-aware",
                      "learned": "learned",
                      "truthful": "truthful (no-reaction)"}[PLANNER]

    


if __name__ == "__main__":

    print(f"Sweet-spot (Prop A.1 + discounted) on pair ({LOC_A}, {LOC_B})")

    results = compute_or_load(use_cache=True)
    plot_unilateral_incentive(results)
