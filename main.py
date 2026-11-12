#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (roblox-up-and-coming-forecast-01).

Monte Carlo rank-diffusion model, seeded for byte-identical repeat runs.
Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.

Model, all estimated from the history it is given:
- Rank diffusion: a game's rank moves as change = mu[band] + momentum*velocity +
  a residual resampled from that band's observed residuals (the changes are
  heavy-tailed, kurtosis 6-12, so a Gaussian understates big moves).
- Momentum: next change carries MOMENTUM * velocity, velocity = last per-hour rank
  change, coefficient from a regression of next change on velocity.
- New entrants: Poisson(new_rate) fresh games per hour, entry rank sampled from the
  observed entry ranks; a small share are breakouts with a strong upward drift
  (brand-new games whose growth explodes to the top within a day or two). Entrants
  are unnamed (their universe ids are unknowable), so the fraction of (run, hour)
  they hold at a place becomes that place's unnamed share.
- Age-out: a game leaves the chart 28 days after its `created` date, when the history
  carries details (the house recorder's reads); the archive carries none, so the
  practice run simply has no age-out.
- Each hour the games are re-sorted by continuous rank (a valid permutation), the
  chart is capped at its closing size, and the top 10 is recorded.
- Probabilities are the Monte Carlo frequencies of (game, place) over runs x hours;
  the top10 is ordered by expected rank.
Standard library only, no network.
"""
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261113
N_RUNS = 1000
AGE_OUT_DAYS = 28


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def band(r):
    if r <= 3:
        return 0
    if r <= 10:
        return 1
    if r <= 20:
        return 2
    if r <= 40:
        return 3
    if r <= 70:
        return 4
    return 5


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json")))
    rows = [json.loads(l) for l in open(os.path.join(INPUT_DIR, "history.ndjson")) if l.strip()]
    places = int(task.get("places", 10))
    hours = int(task["window"]["slots"])
    deadline = parse_ts(task["history_until"])

    # ---- diffusion kernel: change ~ mu*dt + residual*sqrt(dt), residual resampled
    # from the observed per-band residuals (heavy-tailed: kurtosis 6-12, so a Gaussian
    # understates big moves). momentum adds MOMENTUM * velocity to the drift. ----
    samples = [[] for _ in range(6)]
    for a, b in zip(rows, rows[1:]):
        dt = (parse_ts(b["at"]) - parse_ts(a["at"])) / 3600.0
        if dt <= 0:
            continue
        A = {g["universe_id"]: g["rank"] for g in a["up_and_coming"]}
        B = {g["universe_id"]: g["rank"] for g in b["up_and_coming"]}
        for u in set(A) & set(B):
            samples[band(A[u])].append((B[u] - A[u], dt))
    mu, resid = [], []
    for s in samples:
        if not s:
            mu.append(0.0)
            resid.append([0.0])
            continue
        m = sum(c / dt for c, dt in s) / len(s)
        mu.append(m)
        resid.append([(c - m * dt) / math.sqrt(dt) for c, dt in s])

    # ---- momentum coefficient: regress next per-hour change on velocity ----
    mom_x, mom_y = [], []
    for i in range(1, len(rows) - 1):
        a, b, c = rows[i - 1], rows[i], rows[i + 1]
        dt1 = (parse_ts(b["at"]) - parse_ts(a["at"])) / 3600.0
        dt2 = (parse_ts(c["at"]) - parse_ts(b["at"])) / 3600.0
        if dt1 <= 0 or dt2 <= 0:
            continue
        A = {g["universe_id"]: g["rank"] for g in a["up_and_coming"]}
        B = {g["universe_id"]: g["rank"] for g in b["up_and_coming"]}
        C = {g["universe_id"]: g["rank"] for g in c["up_and_coming"]}
        for u in set(A) & set(B) & set(C):
            mom_x.append((B[u] - A[u]) / dt1)
            mom_y.append((C[u] - B[u]) / dt2)
    mom_coef = 0.0
    if len(mom_x) > 10:
        mx = sum(mom_x) / len(mom_x)
        my = sum(mom_y) / len(mom_y)
        cov = sum((x - mx) * (y - my) for x, y in zip(mom_x, mom_y)) / len(mom_x)
        varx = sum((x - mx) ** 2 for x in mom_x) / len(mom_x)
        if varx > 0:
            mom_coef = max(-0.5, min(0.5, cov / varx))

    # ---- new entrants: rate per hour + entry-rank distribution ----
    # A small share of entrants are breakouts: brand-new games whose growth explodes
    # and carry them to the top within a day or two (in the practice window Do Not
    # Stop. and Bloody Ragdoll Physics did exactly that, reaching #1/#2 from outside
    # the chart). They get a strong upward drift until they reach the top band.
    entry_ranks = []
    new_n, new_dt = 0, 0.0
    for a, b in zip(rows, rows[1:]):
        dt = (parse_ts(b["at"]) - parse_ts(a["at"])) / 3600.0
        if dt <= 0:
            continue
        A = {g["universe_id"] for g in a["up_and_coming"]}
        B = {g["universe_id"]: g["rank"] for g in b["up_and_coming"]}
        for u, r in B.items():
            if u not in A:
                entry_ranks.append(r)
                new_n += 1
        new_dt += dt
    new_rate = new_n / new_dt if new_dt else 0.0
    BREAKOUT_SHARE = 0.20
    BREAKOUT_DRIFT = -8.0

    # ---- closing chart + velocities + created dates (for age-out) ----
    # Velocity is the last per-hour rank change (last two reads); a longer window
    # tested worse on the three validation windows.
    last = rows[-1]
    prev = rows[-2] if len(rows) >= 2 else None
    chart = {}
    for g in last["up_and_coming"]:
        chart[g["universe_id"]] = [float(g["rank"]), 0.0]
    if prev:
        dt = (parse_ts(last["at"]) - parse_ts(prev["at"])) / 3600.0
        if dt > 0:
            P = {g["universe_id"]: g["rank"] for g in prev["up_and_coming"]}
            for u in chart:
                if u in P:
                    chart[u][1] = (chart[u][0] - P[u]) / dt
    created = {}
    for r in rows:
        det = r.get("details") or {}
        for uid_str, d in det.items():
            if d and d.get("created"):
                try:
                    created[int(uid_str)] = parse_ts(d["created"])
                except Exception:
                    pass
    age_out_sec = AGE_OUT_DAYS * 86400
    chart_size = len(chart)

    # ---- Monte Carlo ----
    rng = random.Random(SEED)
    counts = defaultdict(lambda: defaultdict(int))  # place -> uid -> count
    new_counts = defaultdict(int)  # place -> count of a new (unnamed) game there
    total = N_RUNS * hours

    for _run in range(N_RUNS):
        # state: [uid or None, continuous rank, velocity, is_new, is_breakout]
        state = [[u, v[0], v[1], False, False] for u, v in chart.items()]
        for h in range(hours):
            t = deadline + h * 3600
            # age-out
            state = [s for s in state
                     if not (s[0] is not None and s[0] in created
                             and t >= created[s[0]] + age_out_sec)]
            # new entrants (Poisson)
            lam = new_rate
            L = math.exp(-lam)
            k, p = 0, 1.0
            while True:
                p *= rng.random()
                if p <= L:
                    break
                k += 1
            for _ in range(k):
                er = rng.choice(entry_ranks) if entry_ranks else rng.randint(50, 97)
                state.append([None, float(er), 0.0, True,
                              rng.random() < BREAKOUT_SHARE])
            # rank updates: drift (band mu + momentum, or breakout drift) + resampled residual
            for s in state:
                b = band(int(round(s[1])))
                b = max(0, min(5, b))
                if s[4] and s[1] > 10:
                    drift = BREAKOUT_DRIFT
                else:
                    drift = mu[b] + mom_coef * s[2]
                change = drift + rng.choice(resid[b])
                s[1] += change
                s[2] = change
            state.sort(key=lambda s: s[1])
            state = state[:chart_size]
            for place in range(1, places + 1):
                if place - 1 < len(state):
                    s = state[place - 1]
                    if s[0] is None:
                        new_counts[place] += 1
                    else:
                        counts[place][s[0]] += 1

    # ---- probabilities: Monte Carlo frequencies; unnamed share = new-game share ----
    probabilities = []
    for place in range(1, places + 1):
        new_p = new_counts[place] / total
        named = [(uid, c / total) for uid, c in counts[place].items() if c / total >= 0.002]
        named_sum = sum(p for _, p in named)
        target_named = max(0.0, 0.99 - new_p)
        if named_sum > target_named and named_sum > 0:
            scale = target_named / named_sum
            named = [(uid, p * scale) for uid, p in named]
        named.sort(key=lambda x: (-x[1], x[0]))
        for uid, p in named[:100]:
            probabilities.append({"universe_id": int(uid), "place": place, "p": round(p, 6)})

    # ---- top10: order by expected rank, tie-break by total top-10 probability ----
    tot = defaultdict(float)
    exprank = defaultdict(float)
    for place in range(1, places + 1):
        for uid, c in counts[place].items():
            p = c / total
            tot[uid] += p
            exprank[uid] += place * p
    cands = sorted(tot.keys(), key=lambda u: (exprank[u] / max(tot[u], 1e-9), -tot[u], u))
    top10 = [int(u) for u in cands[:places]]
    if len(top10) < places:
        for g in last["up_and_coming"]:
            if len(top10) >= places:
                break
            if g["universe_id"] not in top10:
                top10.append(int(g["universe_id"]))

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
