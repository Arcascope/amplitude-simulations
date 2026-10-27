"""Month-by-month circadian phase delay: permanent DST vs permanent standard time.

Both scenarios keep identical clock-based behavior. The only difference is where
the sun sits on the clock:
    ST  scenario: clock = UTC-5 all year (sunrise/sunset in EST)
    DST scenario: clock = UTC-4 all year (sunrise/sunset one clock-hour later)

Uses the Hannay model parameters from simple_melatonin.py, integrated with RK4 at
1-minute resolution. Reports, per month, the model DLMO (clock time at which the
phase crosses the melatonin-onset phase used in simple_melatonin.py) under each
scenario and the DST-minus-ST delay.

Light schedules (--schedule):
    office   Original dst_simulation.py rules: asleep 23:00-07:00; daylight 07:00-09:00
             and 17:00-23:00 whenever the sun is up (100 lux otherwise); 500 lux at
             work 09:00-17:00.
    commute  Wake 06:00, 1-hour commutes 06:30-07:30 and 16:00-17:00 (daylight if the
             sun is up, 50 lux otherwise), work 07:30-16:00 at 500 lux, home at
             300 lux when the sun is up (window light) and 100 lux after dark,
             dimmed to 50 lux 21:30-22:30, asleep 22:30-06:00.

Example:
    python dst_phase_delay_by_month.py --schedule commute --daylight-lux 2000
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sunrise, sunset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_melatonin as sm  # noqa: E402

PHI_ON = np.pi - 6.5 * 2 * np.pi / 24  # melatonin onset phase, as in simple_melatonin.py
MINUTES = np.arange(1440)


def window(h0, h1):
    return (MINUTES >= h0 * 60) & (MINUTES < h1 * 60)


def sun_clock_minutes(observer, day, utc_offset_h):
    """Sunrise and sunset, in minutes after clock midnight, for a fixed UTC offset."""
    tz = timezone(timedelta(hours=utc_offset_h))
    out = []
    for f in (sunrise, sunset):
        t = f(observer, date=day.date(), tzinfo=tz)
        out.append(t.hour * 60 + t.minute + t.second / 60)
    return out


def office_day(sun_up, daylight_lux):
    lux = np.zeros(1440)
    for w in (window(7, 9), window(17, 22 + 59 / 60)):
        lux[w & sun_up] = daylight_lux
        lux[w & ~sun_up] = 100.0
    lux[window(9, 17)] = 500.0
    return lux


def commute_day(sun_up, daylight_lux):
    lux = np.zeros(1440)
    for w in (window(6, 6.5), window(17, 21.5)):  # home: window light when the sun is up
        lux[w & sun_up] = 300.0
        lux[w & ~sun_up] = 100.0
    for w in (window(6.5, 7.5), window(16, 17)):  # commutes
        lux[w & sun_up] = daylight_lux
        lux[w & ~sun_up] = 50.0
    lux[window(7.5, 16)] = 500.0  # work
    lux[window(21.5, 22.5)] = 50.0  # dimmed hour before bed
    return lux


SCHEDULES = {"office": office_day, "commute": commute_day}


def deriv(state, light):
    """Hannay phase-amplitude model, vectorized over scenarios (columns)."""
    R, Psi, n = state
    a = sm.alpha_0 * light ** sm.p / (light ** sm.p + sm.I0)
    Bhat = sm.G * (1.0 - n) * a
    amp = (sm.A1 * 0.5 * Bhat * (1 - R ** 4) * np.cos(Psi + sm.BetaL1)
           + sm.A2 * 0.5 * Bhat * R * (1 - R ** 8) * np.cos(2 * Psi + sm.BetaL2))
    ph = (sm.sigma * Bhat - sm.A1 * Bhat * 0.5 * (R ** 3 + 1 / R) * np.sin(Psi + sm.BetaL1)
          - sm.A2 * Bhat * 0.5 * (1 + R ** 8) * np.sin(2 * Psi + sm.BetaL2))
    return np.stack([
        -sm.gamma * R + sm.K * np.cos(sm.Beta1) / 2 * R * (1 - R ** 4) + amp,
        2 * np.pi / sm.tau + sm.K / 2 * np.sin(sm.Beta1) * (1 + R ** 4) + ph,
        60.0 * (a * (1 - n) - sm.delta * n),
    ])


def dlmo_clock_hours(psi, n_days):
    out = []
    for d in range(n_days):
        seg = np.mod(psi[d * 1440:(d + 1) * 1440] - PHI_ON, 2 * np.pi)
        idx = np.where(np.diff(seg) < -np.pi)[0]  # wrap = crossing PHI_ON
        out.append((idx[0] + 1) / 60 if len(idx) else np.nan)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schedule", choices=SCHEDULES, default="commute")
    parser.add_argument("--daylight-lux", type=float, default=2000.0)
    parser.add_argument("--latitude", type=float, default=42.3601)   # Boston
    parser.add_argument("--longitude", type=float, default=-71.0589)
    parser.add_argument("--year", type=int, default=2024, help="Year to report; the prior half year is spin-up")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    observer = LocationInfo("site", "", "UTC", args.latitude, args.longitude).observer
    days = pd.date_range(datetime(args.year - 1, 7, 1), datetime(args.year, 12, 31), freq="D")
    make_day = SCHEDULES[args.schedule]

    lux = {"ST": [], "DST": []}
    for d in days:
        for label, off in (("ST", -5), ("DST", -4)):
            rise, set_ = sun_clock_minutes(observer, d, off)
            lux[label].append(make_day((MINUTES >= rise) & (MINUTES < set_), args.daylight_lux))
    light = np.stack([np.concatenate(lux["ST"]), np.concatenate(lux["DST"])])

    dt = 1 / 60
    state = np.tile(np.array([1.0, np.pi + np.pi / 12 * 9, 0.519])[:, None], (1, 2))
    psi = np.empty_like(light)
    for i in range(light.shape[1]):
        l = light[:, i]
        k1 = deriv(state, l)
        k2 = deriv(state + dt / 2 * k1, l)
        k3 = deriv(state + dt / 2 * k2, l)
        k4 = deriv(state + dt * k3, l)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        psi[:, i] = state[1]

    df = pd.DataFrame({
        "date": days,
        "dlmo_st": dlmo_clock_hours(psi[0], len(days)),
        "dlmo_dst": dlmo_clock_hours(psi[1], len(days)),
    })
    df["delay_min"] = (df.dlmo_dst - df.dlmo_st) * 60
    df = df[df.date.dt.year == args.year]

    def hhmm(h):
        return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"

    print(f"schedule={args.schedule} daylight_lux={args.daylight_lux:.0f} "
          f"lat={args.latitude} lon={args.longitude} year={args.year}")
    for month, g in df.groupby(df.date.dt.month):
        print(f"{datetime(args.year, month, 1):%b}: DLMO ST {hhmm(g.dlmo_st.mean())}  "
              f"DST {hhmm(g.dlmo_dst.mean())}  delay {g.delay_min.mean():5.1f} min "
              f"(range {g.delay_min.min():.0f}-{g.delay_min.max():.0f})")

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"dst_delay_{args.schedule}_{int(args.daylight_lux)}lux_{args.year}.csv")
    df.to_csv(path, index=False)
    print(f"Daily results written to {path}")


if __name__ == "__main__":
    main()
