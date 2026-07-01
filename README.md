# Adversarial Interaction Attacks in Refugee Matching

Simulation code for the semester thesis *Adversarial Interaction Attacks in
Refugee Matching* (Sophia Lahrech, Automatic Control Laboratory, ETH Zürich).

The project studies what happens when the locations in a learning-based refugee
matching system behave strategically: instead of reporting false data, a location
can genuinely under-support the refugee types it would rather not host, so that
their realised outcomes look worse and the predictor learns to send those types
elsewhere. We model this as a repeated game, characterise its equilibria, and test
two interventions (a rate limit and a minimum-intake quota).

## Repository contents

All code is in `modelling_with_nokarma/`.

**Core model**
- `simu2_real.py` — data loader, matching LP (`solve_lp`), per-individual utility
  (Proposition A.1), attack model, and the three planners (learned / oracle /
  best-response).
- `play_multi_loc_attack.py` — 16-location model loader and the quota-floor LP
  (`solve_lp_with_kappa`).
- `matching_rate_limited.py` — rate-limited matching LP (Intervention 1).
- `ibr_common.py` — iterated best-response helpers on the payoff matrix.

**Experiments (each reproduces figures in the report)**
- `plot_loc_asymmetry_propA1.py` — two-location utility heatmaps (Nash / Pareto).
- `plot_ibr_propA1.py` — IBR trajectory and per-type employment.
- `plot_mpc_br_propA1.py` — MPC best-response comparison.
- `beta_sweep.py` — equilibrium attack level vs. discount factor.
- `plot_sweet_spot_propA1.py` — rate-limit intervention sweep.
- `play_multi_loc_step1.py` / `step2.py` / `step3.py` — single attacker, two
  attackers, and quota-floor experiments in the 16-location setting.
- `async_ibr_16loc.py` — asynchronous best response over all 16 locations
  (Appendix A).

## Data

The simulations are calibrated on real HIAS resettlement data, which is **not**
included in this repository for confidentiality reasons. The scripts expect the
calibrated employment probabilities and capacity files to be available locally
(loaded via the paths in `simu2_real.py`). Without them the code will not run, but
it fully documents the method.

## Requirements

Python 3, with `numpy`, `pandas`, `scipy`, `scikit-learn`, and `matplotlib`.

## Reference

See the accompanying report for the full model, notation, and results.
