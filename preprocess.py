"""
Preprocessing pipeline for the mcPHASES menstrual-tracking dataset.

Reads the raw CSVs in the repo root and produces two model-ready outputs:
  - master_daily.csv : one row per (id, study_interval, day_in_study) with
                       symptom/hormone/Fitbit features + daily regression targets
                       (days_to_next_period, is_bleeding).
  - cycles.csv       : one row per detected menstrual cycle with cycle_length,
                       period_duration, and baseline covariates — use this for
                       cycle-level models.

Run:  python preprocess.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
# Raw CSVs live one level up (dataset dir). Outputs are written alongside
# this script inside period-tracker/.
DATA_DIR = HERE.parent
OUT_DIR = HERE

# ------------------------------------------------------------------ constants

LIKERT_MAP = {
    "Not at all": 0,
    "Very Low/Little": 1,
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5,
}

# flow_volume is an 8-level ordinal (Not at all ... Very Heavy)
FLOW_MAP = {
    "Not at all": 0,
    "Spotting / Very Light": 1,
    "Very Light": 1,
    "Somewhat Light": 2,
    "Light": 3,
    "Moderate": 4,
    "Somewhat Heavy": 5,
    "Heavy": 6,
    "Very Heavy": 7,
}

PHASE_MAP = {"Menstrual": 0, "Follicular": 1, "Fertility": 2, "Luteal": 3}

LIKERT_COLS = [
    "appetite", "exerciselevel", "headaches", "cramps", "sorebreasts",
    "fatigue", "sleepissue", "moodswing", "stress", "foodcravings",
    "indigestion", "bloating",
]

# gap (days) between bleeding-days that still counts as the SAME period
PERIOD_GAP_TOLERANCE = 2
# window (days) with no bleeding required before a new period_start is declared
NEW_CYCLE_QUIET_WINDOW = 7


# --------------------------------------------------------------- load helpers

def _read(name: str) -> pd.DataFrame:
    # Prefer the raw-dataset directory; fall back to period-tracker/ for
    # files that also live here (e.g. sleep_score.csv).
    p = DATA_DIR / name
    if not p.exists():
        p = HERE / name
    return pd.read_csv(p)


def load_self_report() -> pd.DataFrame:
    df = _read("hormones_and_selfreport.csv")
    # encode flow/phase
    df["flow_volume_num"] = df["flow_volume"].map(FLOW_MAP)
    df["flow_color_num"] = df["flow_color"].map(FLOW_MAP)  # same ordinal scale-ish; NaN for unknown
    df["phase_num"] = df["phase"].map(PHASE_MAP)
    # encode Likert symptoms
    for c in LIKERT_COLS:
        if c in df.columns:
            df[c + "_num"] = df[c].map(LIKERT_MAP)
    return df


# ------------------------------------------------------ cycle/target derivation

def derive_cycle_labels(sr: pd.DataFrame) -> pd.DataFrame:
    """Add is_bleeding, period_start, cycle_id, day_in_cycle, days_to_next_period."""
    df = sr.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)

    df["is_bleeding"] = (df["flow_volume_num"].fillna(0) > 0).astype(int)

    # A period_start requires is_bleeding=1 AND no bleeding in the prior
    # NEW_CYCLE_QUIET_WINDOW days (within same id+interval). This filters out
    # intra-period "Not at all" gaps incorrectly flipping to a new start.
    def _flag_starts(g: pd.DataFrame) -> pd.Series:
        bleed = g["is_bleeding"].values
        day = g["day_in_study"].values
        starts = np.zeros(len(g), dtype=int)
        last_bleed_day = -10_000
        for i in range(len(g)):
            if bleed[i] == 1:
                if day[i] - last_bleed_day > NEW_CYCLE_QUIET_WINDOW:
                    starts[i] = 1
                last_bleed_day = day[i]
        return pd.Series(starts, index=g.index)

    df["period_start"] = (
        df.groupby(["id", "study_interval"], group_keys=False).apply(_flag_starts)
    )

    # cycle_id = cumulative count of period_starts within (id, interval).
    df["cycle_id"] = df.groupby(["id", "study_interval"])["period_start"].cumsum()

    # day_in_cycle = days since last period_start; NaN before the first one
    def _day_in_cycle(g: pd.DataFrame) -> pd.Series:
        start_day = g.loc[g["period_start"] == 1, "day_in_study"]
        if start_day.empty:
            return pd.Series([np.nan] * len(g), index=g.index)
        # for each row, find the most recent period_start day <= current day
        days = g["day_in_study"].values
        starts_sorted = np.sort(start_day.values)
        idx = np.searchsorted(starts_sorted, days, side="right") - 1
        out = np.where(idx >= 0, days - starts_sorted[np.clip(idx, 0, None)], np.nan)
        return pd.Series(out, index=g.index)

    df["day_in_cycle"] = df.groupby(["id", "study_interval"], group_keys=False).apply(_day_in_cycle)

    # days_to_next_period = days until next period_start (>= current day);
    # NaN if no subsequent period_start in the same interval (right-censored).
    def _days_to_next(g: pd.DataFrame) -> pd.Series:
        days = g["day_in_study"].values
        start_days = np.sort(g.loc[g["period_start"] == 1, "day_in_study"].values)
        if len(start_days) == 0:
            return pd.Series([np.nan] * len(g), index=g.index)
        idx = np.searchsorted(start_days, days, side="left")
        out = np.where(idx < len(start_days), start_days[np.clip(idx, 0, len(start_days) - 1)] - days, np.nan)
        return pd.Series(out, index=g.index)

    df["days_to_next_period"] = df.groupby(["id", "study_interval"], group_keys=False).apply(_days_to_next)

    return df


def build_cycles_table(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle: cycle_length, period_duration, start-day covariates."""
    rows = []
    for (pid, interval), g in daily.groupby(["id", "study_interval"]):
        g = g.sort_values("day_in_study").reset_index(drop=True)
        start_idx = g.index[g["period_start"] == 1].tolist()
        for n, si in enumerate(start_idx):
            start_day = int(g.loc[si, "day_in_study"])
            # cycle_length = days until next period_start (NaN for last cycle)
            if n + 1 < len(start_idx):
                next_start = int(g.loc[start_idx[n + 1], "day_in_study"])
                cycle_length = next_start - start_day
            else:
                next_start = np.nan
                cycle_length = np.nan

            # period_duration = consecutive bleeding days starting at this start,
            # merging gaps <= PERIOD_GAP_TOLERANCE.
            bleed_days = g.loc[g["is_bleeding"] == 1, "day_in_study"].values
            bleed_in_cycle = bleed_days[
                (bleed_days >= start_day)
                & (bleed_days < (next_start if not np.isnan(next_start) else 10**9))
            ]
            if len(bleed_in_cycle) == 0:
                duration = np.nan
            else:
                duration = 1
                prev = bleed_in_cycle[0]
                for d in bleed_in_cycle[1:]:
                    if d - prev <= PERIOD_GAP_TOLERANCE + 1:
                        duration = int(d - start_day + 1)
                        prev = d
                    else:
                        break

            rows.append({
                "id": pid,
                "study_interval": interval,
                "cycle_index": n + 1,
                "start_day_in_study": start_day,
                "next_start_day_in_study": next_start,
                "cycle_length": cycle_length,
                "period_duration": duration,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- fitbit daily merge

def _daily_agg(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    """Aggregate rows to one per (id, study_interval, day_in_study) via mean."""
    keep = ["id", "study_interval", "day_in_study"] + value_cols
    df = df[keep].copy()
    out = df.groupby(["id", "study_interval", "day_in_study"], as_index=False)[value_cols].mean()
    out = out.rename(columns={c: f"{prefix}_{c}" for c in value_cols})
    return out


def build_daily_fitbit() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # resting heart rate
    try:
        rhr = _read("resting_heart_rate.csv")
        frames.append(_daily_agg(rhr, ["value"], "rhr").rename(columns={"rhr_value": "rhr"}))
    except FileNotFoundError:
        pass

    # active minutes
    try:
        am = _read("active_minutes.csv")
        frames.append(_daily_agg(am, ["sedentary", "lightly", "moderately", "very"], "active"))
    except FileNotFoundError:
        pass

    # stress score
    try:
        ss = _read("stress_score.csv")
        frames.append(_daily_agg(ss, ["stress_score"], "fitbit"))
    except FileNotFoundError:
        pass

    # sleep score
    try:
        sleep = _read("sleep_score.csv")
        frames.append(_daily_agg(
            sleep,
            ["overall_score", "composition_score", "revitalization_score",
             "duration_score", "deep_sleep_in_minutes", "resting_heart_rate",
             "restlessness"],
            "sleep",
        ))
    except FileNotFoundError:
        pass

    # respiratory rate summary (timestamp day)
    try:
        rr = _read("respiratory_rate_summary.csv")
        frames.append(_daily_agg(
            rr,
            ["full_sleep_breathing_rate", "deep_sleep_breathing_rate",
             "rem_sleep_breathing_rate"],
            "resp",
        ))
    except FileNotFoundError:
        pass

    # heart-rate zones
    try:
        hrz = _read("time_in_heart_rate_zones.csv")
        frames.append(_daily_agg(
            hrz,
            ["below_default_zone_1", "in_default_zone_1",
             "in_default_zone_2", "in_default_zone_3"],
            "hrz",
        ))
    except FileNotFoundError:
        pass

    # computed temperature — uses sleep_end_day_in_study as the daily key
    try:
        temp = _read("computed_temperature.csv")
        temp = temp.rename(columns={"sleep_end_day_in_study": "day_in_study"})
        frames.append(_daily_agg(
            temp,
            ["nightly_temperature", "baseline_relative_nightly_standard_deviation"],
            "temp",
        ))
    except FileNotFoundError:
        pass

    if not frames:
        return pd.DataFrame(columns=["id", "study_interval", "day_in_study"])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["id", "study_interval", "day_in_study"], how="outer")
    return out


# ---------------------------------------------------------------- subject info

def load_subject_info() -> pd.DataFrame:
    s = _read("subject-info.csv")
    # numeric encoding of literacy (ordinal)
    lit_map = {"Non-existent": 0, "Low": 1, "Medium": 2, "High": 3, "Expert": 4}
    s["literacy_num"] = s["self_report_menstrual_health_literacy"].map(lit_map)
    s["sexually_active_num"] = (s["sexually_active"].str.lower() == "yes").astype("Int64")
    return s[["id", "birth_year", "age_of_first_menarche", "literacy_num", "sexually_active_num"]]


# ---------------------------------------------------------------------- main

def build_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    sr = load_self_report()
    daily = derive_cycle_labels(sr)

    # drop the raw categorical Likert columns — keep only *_num
    drop_cols = [c for c in LIKERT_COLS if c in daily.columns]
    daily = daily.drop(columns=drop_cols + ["flow_volume", "flow_color", "phase"])

    fit = build_daily_fitbit()
    daily = daily.merge(fit, on=["id", "study_interval", "day_in_study"], how="left")

    subj = load_subject_info()
    daily = daily.merge(subj, on="id", how="left")
    # current age at study interval
    daily["age_at_interval"] = daily["study_interval"] - daily["birth_year"]

    # forward-fill slowly-changing physiological signals within (id, interval),
    # max 3 days, so the model sees the last known value rather than NaN.
    ffill_cols = [c for c in daily.columns if c.startswith(("rhr", "sleep_", "temp_", "resp_"))]
    daily[ffill_cols] = (
        daily.sort_values(["id", "study_interval", "day_in_study"])
             .groupby(["id", "study_interval"])[ffill_cols]
             .ffill(limit=3)
    )

    cycles = build_cycles_table(daily)
    # attach subject baseline to cycles too
    cycles = cycles.merge(subj, on="id", how="left")
    cycles["age_at_interval"] = cycles["study_interval"] - cycles["birth_year"]

    return daily, cycles


def main() -> None:
    daily, cycles = build_master()

    out_daily = OUT_DIR / "master_daily.csv"
    out_cycles = OUT_DIR / "cycles.csv"
    daily.to_csv(out_daily, index=False)
    cycles.to_csv(out_cycles, index=False)

    print(f"[ok] master_daily.csv   rows={len(daily):>6}  cols={daily.shape[1]}")
    print(f"[ok] cycles.csv         rows={len(cycles):>6}  cols={cycles.shape[1]}")
    print()
    print("period_start events per participant:")
    print(daily.groupby("id")["period_start"].sum().describe().round(2).to_string())
    print()
    print("cycle_length (days) summary:")
    print(cycles["cycle_length"].describe().round(2).to_string())
    print()
    print("period_duration (days) summary:")
    print(cycles["period_duration"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
