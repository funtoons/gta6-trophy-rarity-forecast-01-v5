# GTA VI Trophy Rarity Forecast v5

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01-v5/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01-v5/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun GTA VI trophy rarity forecast](https://make.fun/competitions/gta6-trophy-rarity-forecast-01) competition: forecast every PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles, scored by mean pinball loss on the logit scale.

## Model

A launch-curve / dilution model, built from counts, not from the last rate alone. A trophy's rate is earners / players-tracked, and Exophase refreshes its counts in jumps, so a rate moves for two reasons: earners grow as players play (up) and the tracked-player base jumps at a refresh (down).

- **Player growth `G`**: smooth log-linear trend of players-tracked over the recent window, floored by a refresh-hazard term. When the base has been stale, a jump inside the horizon is likely; its expected size comes from the jumps the history already shows (or a launch-night prior when none shows).
- **Dilution per tier**: the median is `E_target / P_target`, with newly counted players earning at `a = p_last * c`. The share `c` comes from the trophy's class — tier/points, secret flag, description wording (chapter / mission / grind), current rate — discounted by how long the newly counted cohort has had to play (about half the stale window plus the horizon).
- **Earner growth (ODE)**: earners chase a moving ceiling, `dE/dt = k * (ceiling*P - E)`, with `k` fitted from the trophy's own history and the ceiling from its class. The ODE forecast and the dilution-aware count forecast estimate the same quantity by different routes; the larger is used.
- **Horizon-aware**: dilution strength and cohort play time scale with the horizon, so a 90-minute practice case and a 7-day final case are not treated alike.
- **Spread**: logit-normal around the median — random walk, dilution uncertainty on the *uncertain* refresh-jump part of the growth, small-count term — with the lower tail widened asymmetrically while the dilution signal is live.

Python 3.13, standard library only, offline, deterministic, no state between runs. One entrypoint: `python3 main.py` reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Validation (synthetic worlds, not the practice answers)

The practice answers live server-side, so the model was tuned only on synthetic worlds built from first principles (players arrive smoothly, Exophase displays counts at refreshes, earners follow a cohort ODE), with each trophy's truth curve solved to reproduce the real launch-night rates. Mean pinball loss per scenario (house baseline in brackets):

| scenario | v5 | baseline |
|---|---|---|
| normal, 90 min | 0.071 | 0.038 |
| late flood x3, 90 min | 0.312 | 0.882 |
| late flood x6, 90 min (practice proxy) | 0.306 | 1.520 |
| late flood x6, easiness shuffled | 0.307 | 1.520 |
| uniform dilution, 90 min | 0.211 | 1.418 |
| final-like, 4.5-day horizon, steady | 0.194 | 0.242 |
| final-like + unseen surge x2 | 0.370 | 0.323 |

Practice (p00/p01) runs produce valid output: 55 rows, five non-decreasing quantiles in [0, 100], ~2.8 KB per file. Two runs are byte-identical; runtime is ~0.08 s for both practice cases (limit 8 s per case). Edge cases verified: single read, empty history, null rates, 0%/100% rates, 1-minute and 30-day horizons.

Honest limits: the 0.306 practice figure is a *simulated proxy*, not the real practice loss (the real answers are server-side). The model still loses to the house baseline when no refresh happens at all (0.071 vs 0.038) and when a surge lands that no history could foresee (0.370 vs 0.323).

## License

MIT — see [LICENSE](LICENSE).
