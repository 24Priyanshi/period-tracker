# Period Prediction — Approach Document

## Goal
Build a model that predicts a participant's **next menstrual period date range** (start day and expected duration) using a combination of:

- Self-reported symptoms & menstrual flow logs
- Hormone measurements (LH, estrogen metabolite, PDG) from Mira device
- Fitbit physiological signals (resting HR, HRV, sleep, temperature, respiratory rate, stress, activity)

## Dataset Overview (mcPHASES)
- **42 participants**, 2 study intervals: 2022 (Jan–Apr) and 2024 (Jul–Oct)
- **~5,659 participant-days** of labelled self-report / hormone data
- Flow volume and cycle-phase labels are available on most days, which lets us derive ground-truth period-start events
- Raw cycle-length distribution is noisy (median ~12 days) because bleeding days are often broken up by a "Not at all" day — we correct this in preprocessing

## Problem Framing
We convert this into two complementary supervised learning problems:

### Target 1 — Cycle-length regression (primary)
For each detected period start, predict the number of days until the **next** period start.
- Target: `cycle_length` (days)
- One row per cycle per participant (per interval)

### Target 2 — Daily "days-until-next-period" regression (richer signal)
For every day the participant has data, predict **`days_to_next_period`**.
- Target: continuous (e.g. 0 if today is period day 1, 14 if period starts in 2 weeks)
- Lets us train on all days, not just cycle-start events

### Target 3 — Period duration regression
Predict how many consecutive bleeding days a period will last.
- Target: `period_duration`

Final deliverable = model that, given a participant's recent signals + last known period start, outputs **(predicted_next_start_day, predicted_duration)** → the requested **period date range**.

## Key Preprocessing Steps

1. **Per-interval isolation**: 2022 and 2024 are separate time series per participant; never compute cycle metrics across intervals.

2. **Flow → bleeding flag**: `is_bleeding = flow_volume not in {NaN, "Not at all"}`.

3. **Period episode detection** (robust period_start):
   - A day is a `period_start` if `is_bleeding=1` AND no bleeding in the prior **7 days** within the same participant + interval.
   - Prevents treating an intra-period "Not at all" day as a new cycle start.

4. **Cycle derivation**:
   - `cycle_id` = running count of period episodes per (id, interval)
   - `day_in_cycle` = days since last period_start
   - `cycle_length` = days between consecutive period_starts (assigned to cycle start row)
   - `period_duration` = consecutive bleeding-days count for each episode

5. **Categorical encoding** of Likert symptom columns to 0–5 numeric scale (consistent across all symptoms):
   `Not at all → 0, Very Low/Little → 1, Low → 2, Moderate → 3, High → 4, Very High → 5`
   `flow_volume` uses an 8-level ordinal 0–7 scale.

6. **Daily feature merge** — join per-day Fitbit signals onto the (id, study_interval, day_in_study) grain:
   - `resting_heart_rate.value` → `rhr`
   - `sleep_score` → overall_score, composition, revitalization, duration, deep_sleep_minutes, restlessness
   - `active_minutes` → sedentary, lightly, moderately, very
   - `stress_score` → stress_score
   - `respiratory_rate_summary` → full_sleep_breathing_rate
   - `time_in_heart_rate_zones` → below_default_zone_1, in_default_zone_1/2/3
   - `computed_temperature` (mapped via sleep_end_day) → nightly_temperature, baseline_relative_nightly_sd
   - `subject-info` (static) → birth_year, age_of_menarche, literacy

7. **Missing data**: per-column forward-fill within (id, interval) for slowly-changing physiological signals (e.g., RHR) up to 3 days; leave NaN elsewhere so the model can handle/impute.

## Output Artifacts

| File | Purpose |
|------|---------|
| `preprocess.py` | End-to-end pipeline — reads raw CSVs, produces clean outputs |
| `master_daily.csv` | One row per (id, interval, day_in_study) — features + daily targets |
| `cycles.csv` | One row per detected cycle — cycle_length, period_duration, baseline covariates |
| `DATASET.md` | Schema + column descriptions for downstream modeling |

## Recommended Modeling Track (next phase)

1. **Baselines**
   - Predict cycle length = participant's historical mean cycle length
   - Predict next period = last period_start + 28
2. **Classical ML**
   - Gradient Boosted Trees (LightGBM/XGBoost) on `cycles.csv` for cycle_length regression
   - Same on `master_daily.csv` for `days_to_next_period` regression
3. **Sequence models** (if enough data per participant)
   - LSTM / Temporal CNN over last N days of physiological signals → next period start
4. **Per-participant personalization**
   - Include participant id as a categorical / embedding, or fit hierarchical model

Evaluation: leave-one-cycle-out CV per participant; report MAE in days for cycle_length and period_start prediction.
