# Roblox Up-and-Coming Top-10 Forecaster

Entry for make.fun `roblox-up-and-coming-forecast-01` ($500): a program that reads
the history of Roblox's Up-and-Coming chart and forecasts its top 10, with a
probability for each game at each place, for the 72 hours after entries close.

## How it works

A seeded Monte Carlo rank-diffusion model, all parameters estimated from the
history it is given (`/input/history.ndjson`):

- **Rank diffusion.** A game's rank moves as `mu[band] + momentum * velocity +
  residual`, where the residual is resampled from that rank band's observed
  residuals. Rank changes are heavy-tailed (kurtosis 6-12), so a Gaussian
  understates big moves; resampling the empirical residuals keeps them.
- **Momentum.** The next change carries a fraction of the game's last per-hour
  rank change (coefficient from a regression of next change on velocity).
- **New entrants.** Fresh games arrive as a Poisson process at the observed rate,
  with entry ranks sampled from the observed entry ranks. A small share are
  *breakouts*: brand-new games whose growth explodes and carry them to the top
  within a day or two (the practice window had two: Do Not Stop. and Bloody
  Ragdoll Physics reached #1/#2 from outside the chart). Entrants are unnamed —
  their universe ids are unknowable — so the share of (run, hour) they hold at a
  place becomes that place's unnamed share.
- **Age-out.** A game leaves the chart 28 days after its `created` date, when the
  history carries details (the house recorder's reads); the public archive
  carries none, so the practice run simply has no age-out.
- Each hour the games are re-sorted by continuous rank (a valid permutation),
  the chart is capped at its closing size, and the top 10 is recorded.
- Probabilities are the Monte Carlo frequencies of (game, place) over runs x
  hours; the top 10 is ordered by expected rank.

## Validation (local, practice window)

Scored against the 10 archive reads in the practice window
(2026-11-07T23:00Z to 2026-11-10T23:00Z), reconstructed from the public
archive `Kenny-Dhub/roblox-snapshots`:

| forecast | log score | typical p/place |
|---|---:|---:|
| house baseline (persistence) | -43.512039 | 0.012891 |
| this forecaster | **-34.049980** | 0.033207 |

Parameters were chosen by the mean log score over three 72-hour windows cut
from the history (2026-11-02, 2026-11-04, 2026-11-07), not by the single
official practice window, so the tuning is not overfit to it.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic: the RNG is
seeded, so two runs on the same history write byte-identical output
(~9 s for 1000 runs x 72 hours on 1 CPU).

## License

MIT (see LICENSE).
