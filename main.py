"""GTA VI trophy rarity forecast v5 (Python 3.13, standard library only).

Model, from first principles (launch-curve / dilution model):

A trophy's earn rate is earners / players-tracked.  Exophase refreshes its
counts in jumps, so a rate moves for two separate reasons: earners grow as
players play (trend, up) and the tracked-player base jumps at a refresh
(dilution, down).  The forecast therefore builds the median from counts,
not from the last rate alone:

    median = E_target / P_target

    P_target = P_last * G,  G from the tracked-player trend in the case's
        history, with a refresh-hazard term when the base has been stale:
        the longer the base has not moved, the likelier a jump lands inside
        the horizon, and the jump's expected size comes from the jumps the
        history already shows (or a launch-night prior when none shows).
    E_target = max(E_ode, E_count)
        E_ode: earners under dE/dt = k * (ceiling*P - E), with k fitted
            from the trophy's own history and ceiling from its class --
            earners chase a moving ceiling as the player base grows;
        E_count: E_last + (P_target - P_last) * a, the refresh cohort's
            head start -- a = p_last * c, the share of newly counted
            players who have earned the trophy, discounted by how long
            that cohort has had to play (about half the stale window
            plus the horizon: a refresh reveals players who arrived
            over the stale period, not players who have played since
            release).  Each route is blind to the other's part, so the
            larger of the two is used.

Class easiness comes only from the case itself: tier / points, secret
flag, the trophy's description (chapter / mission / grind wording) and
its current rate.  Earner growth is not assumed from the class: the
ODE rate k is fitted from the trophy's own history, so a flat
launch-night history gives k = 0 and a moving two-day history gives
the growth it actually shows.

Spread: logit-normal around the median with a random-walk term, a
dilution-uncertainty term that grows with the log of the *uncertain*
refresh-jump part of the growth (not the smooth trend the history
already shows), and an asymmetric lower widening while the dilution
signal is live (the refresh branch sits far below the no-refresh
branch, so the low quantiles must reach it).

Deterministic, offline, no state between runs: every input is read from
the case file, every output is a pure function of it.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Tunable shape constants (calibrated on synthetic worlds, not on any
# --- published answer: see the validation notes in the repository README).
SIGMA_RW = 0.4        # logit random-walk spread per sqrt(day)
SIGMA_FLOOR = 0.30    # minimum spread: never claim false precision
K_DIL = 0.4           # dilution-uncertainty weight on log(refresh jump)
ASYM_LOW = 0.40       # extra lower-tail widening while dilution is live
ASYM_HIGH = 0.15      # mild upper-tail widening
JUMP_PRIOR = 6.0      # expected size of one launch-night refresh jump
GAP_PRIOR_H = 4.0     # typical hours between refreshes when none observed
G_MAX = 50.0          # cap on predicted player-base growth


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def pct_to_frac(pct):
    return clamp(pct, 0.01, 100.0) / 100.0


def logit_frac(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 (with Z or offset, date-only allowed) -> aware datetime."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_for(level):
    for known, z in zip(LEVELS, Z):
        if abs(level - known) < 1e-9:
            return z
    return norm_ppf(clamp(level, 0.001, 0.999))


# --- Trophy class: how quickly a fresh player earns it ---------------------

GRIND_WORDS = ("100%", "every", "all ", "collect", "250 ", "50 ",
               " 8 ", " 3 ", "20 ", "10 ", "earn $", "spend $",
               "purchase all", "register", "locations", "bank vaults",
               "without being", "gold medal", "complete 8", "complete 3",
               "complete 50", "complete 250")
EASY_WORDS = ("beat ", "discover", "try each", "take down", "lose a ",
              "complete an arcade", "enjoy a cocktail", "bowl 'em",
              "follow-through")


def easiness(trophy, rate_frac):
    """0..1: how attainable the trophy is for a fresh player.

    From the case only: tier, description wording, current rate.
    """
    tier = (trophy.get("tier") or "bronze").lower()
    base = {"platinum": 0.05, "gold": 0.15, "silver": 0.30,
            "bronze": 0.55}.get(tier, 0.45)
    desc = (trophy.get("description") or "").lower()
    if "complete chapter" in desc:
        e = 0.90
    elif desc.startswith("complete the final"):
        e = 0.15
    elif any(w in desc for w in GRIND_WORDS):
        e = 0.15
    elif desc.startswith("complete "):
        e = 0.55
    elif any(w in desc for w in EASY_WORDS):
        e = 0.70
    else:
        e = base
    if trophy.get("secret"):
        e *= 0.80
    # A trophy almost everyone already holds is earned fast by newcomers;
    # a trophy almost nobody holds is not.
    e *= 0.60 + 0.40 * (rate_frac / (rate_frac + 0.008))
    e = clamp(e, 0.03, 0.95)
    if rate_frac >= 0.5:
        e = 0.98
    return e


def attainment_share(e, play_days):
    """c: the share of the current rate at which newly counted players
    have earned a trophy of easiness e, given the cohort's average play
    time.  Easy trophies are earned within the first hour of play (short
    time constant), so c reaches e fast; grind trophies need days, so c
    stays near zero over a short window."""
    tau = 0.03 + 1.0 * (1.0 - e) * (1.0 - e)
    return e * (1.0 - math.exp(-play_days / tau))


def fit_log_slope(points):
    """Least-squares slope of log(value) against hours; 0 when the
    series is flat or too short to fit."""
    if len(points) < 3:
        return 0.0
    xs = [p[0] for p in points]
    ys = [math.log(p[1]) for p in points]
    fit = linfit(xs, ys)
    return fit[0] if fit is not None else 0.0


def ode_earners(e0, p0, s_p, horizon_h, k, ceiling):
    """Earners at the target under dE/dt = k * (ceiling*P(t) - E),
    with P(t) = p0 * exp(s_p * t): earners chase a moving ceiling as
    the player base grows.  Closed form; k = 0 returns e0."""
    if k <= 0.0 or horizon_h <= 0.0:
        return e0
    if s_p <= 0.0:
        return ceiling * p0 + (e0 - ceiling * p0) * math.exp(-k * horizon_h)
    decay = math.exp(-k * horizon_h)
    growth = math.exp(s_p * horizon_h)
    return (e0 * decay
            + k * ceiling * p0 * (growth - decay) / (k + s_p))


# --- History fits -----------------------------------------------------------

def linfit(xs, ys):
    """Least-squares slope/intercept; None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def growth_factor(history, at_dt, horizon_h, release_dt):
    """Predicted P_target / P_last.

    Smooth log-linear trend of players-tracked over the recent window,
    floored by a refresh-hazard term: when the base has been stale, a
    jump inside the horizon is likely and the expected jump is sized
    from the jumps the history already shows.
    """
    pts = []
    for h in history:
        t = parse_time(h.get("at"))
        p = h.get("players_tracked")
        if t is not None and isinstance(p, (int, float)) and p > 0:
            pts.append((t, float(p)))
    if not pts:
        return 1.0, 0.0, 0.0, 1.0
    pts.sort(key=lambda tp: tp[0])
    last_p = pts[-1][1]
    t0 = pts[0][0]
    # Recent window: last 24 h of reads, else everything.
    recent = [tp for tp in pts
              if (at_dt - tp[0]).total_seconds() <= 24 * 3600] or pts
    xs = [(tp[0] - t0).total_seconds() / 3600.0 for tp in recent]
    ys = [math.log(tp[1]) for tp in recent]
    fit = linfit(xs, ys)
    g_smooth = 1.0
    if fit is not None:
        slope = fit[0]
        if slope > 0:
            g_smooth = math.exp(slope * horizon_h)
    # Refresh hazard from staleness and observed jump sizes.
    jumps = []
    prev = None
    last_change = pts[0][0]
    gaps = []
    for t, p in pts:
        if prev is not None and p != prev[1]:
            if p > prev[1]:
                jumps.append(p / prev[1])
            gaps.append((t - last_change).total_seconds() / 3600.0)
            last_change = t
        prev = (t, p)
    staleness = (at_dt - last_change).total_seconds() / 3600.0
    tau_r = clamp(sorted(gaps)[len(gaps) // 2] if gaps else GAP_PRIOR_H,
                  1.0, 24.0)
    jump = sorted(jumps)[len(jumps) // 2] if jumps else JUMP_PRIOR
    jump = clamp(jump, 1.05, 12.0)
    hazard = 1.0 - math.exp(-(staleness + horizon_h) / tau_r)
    g_refresh = 1.0 + hazard * (jump - 1.0)
    g = max(g_smooth, g_refresh)
    return clamp(g, 1.0, G_MAX), math.log(g), staleness, g_refresh


# --- Forecast ---------------------------------------------------------------

def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    at_dt = parse_time(case.get("at"))
    target_dt = parse_time(case.get("target"))
    release_dt = parse_time((case.get("game") or {}).get("release_date"))
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0
    if at_dt is None:
        if history:
            at_dt = parse_time(history[-1].get("at"))
        if at_dt is None:
            at_dt = target_dt or release_dt
        if at_dt is None:
            # Degenerate case with no time anywhere: epoch, so the
            # output stays a deterministic function of the input.
            at_dt = datetime.fromtimestamp(0, timezone.utc)
    if target_dt is None:
        target_dt = at_dt
    if release_dt is None:
        release_dt = at_dt

    g, log_g, staleness_h, g_refresh = growth_factor(history, at_dt,
                                                    horizon_h, release_dt)
    # Average play time of the newly counted cohort: players revealed by
    # the next refresh arrived over the stale window plus the horizon.
    play_days = max(0.0, (staleness_h + horizon_h) / 2.0 / 24.0)
    # Log player-base slope over the recent window (shared by trophies).
    p_pts = []
    for h in history:
        t = parse_time(h.get("at"))
        p = h.get("players_tracked")
        if (t is not None and isinstance(p, (int, float)) and p > 0
                and (at_dt - t).total_seconds() <= 24 * 3600):
            p_pts.append(((t - at_dt).total_seconds() / 3600.0, float(p)))
    s_p = fit_log_slope(p_pts)

    # Per-trophy series for the uptake decomposition, and the last
    # non-null rate / player count.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        e_series = []
        p_series = []
        for h in history:
            rates = h.get("rates") or []
            p = h.get("players_tracked")
            if (i < len(rates) and rates[i] is not None
                    and isinstance(p, (int, float)) and p > 0):
                t = parse_time(h.get("at"))
                r = pct_to_frac(float(rates[i]))
                if t is not None:
                    hours = (t - at_dt).total_seconds() / 3600.0
                    if hours >= -24.0:
                        e_series.append((hours, max(r * float(p), 0.01)))
                        p_series.append((hours, float(p)))
                rate = r
        last_rates.append((rate, e_series, p_series))
    last_p = None
    for h in reversed(history):
        p = h.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            last_p = float(p)
            break
    if last_p is None:
        last_p = 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate, e_series, p_series = last_rates[i]
        if rate is None:
            rate = 0.01
        e = easiness(trophy, rate)
        # A trophy (nearly) everyone already holds is earned by
        # essentially every new player too, whatever the play window.
        if rate >= 0.5:
            a = rate * 0.999
        else:
            a = rate * attainment_share(e, play_days)
        earners = rate * last_p
        # Earner-growth rate k, fitted from this trophy's own history:
        # dE/dt = k * (ceiling*P - E), ceiling from the trophy's class.
        # The ODE forecast and the dilution-aware count forecast estimate
        # the same quantity by different routes (the ODE sees the smooth
        # trend, the count form sees the refresh cohort's head start);
        # take the larger, since each route is blind to the other's part.
        s_e = fit_log_slope(e_series)
        headroom = e * last_p - earners
        k = clamp(s_e * earners / headroom, 0.0, 0.02) if headroom > 0 else 0.0
        e_ode = ode_earners(earners, last_p, max(s_p, 0.0), horizon_h, k, e)
        e_count = earners + last_p * (g - 1.0) * a
        e_target = max(e_ode, e_count)
        p_target = last_p * g
        median = clamp(e_target / p_target, 0.0001, 0.9999)
        med_logit = logit_frac(median)

        # Spread: random walk + dilution uncertainty + small-count term,
        # lower tail widened asymmetrically while dilution is live.
        # Dilution uncertainty scales with the *uncertain* part of the
        # growth (the refresh jump), not the smooth trend the history
        # already shows.
        sigma_rw = SIGMA_RW * math.sqrt(max(horizon_d, 0.0))
        sigma_dil = K_DIL * math.log(g_refresh)
        sigma_cnt = 0.30 / math.sqrt(max(earners, 1.0))
        sigma = math.sqrt(sigma_rw ** 2 + sigma_dil ** 2 +
                          SIGMA_FLOOR ** 2 + sigma_cnt ** 2)
        sigma = clamp(sigma, 0.30, 2.2)
        dil_live = clamp(math.log(g_refresh) / 1.8, 0.0, 1.0)
        sigma_low = sigma * (1.0 + ASYM_LOW * dil_live)
        sigma_high = sigma * (1.0 + ASYM_HIGH * dil_live)

        qs = []
        for level in levels:
            z = z_for(level)
            s = sigma_low if z < 0 else sigma_high if z > 0 else sigma
            q = sigmoid(med_logit + z * s) * 100.0
            qs.append(round4(clamp(q, 0.01, 100.0)))
        # Enforce non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
