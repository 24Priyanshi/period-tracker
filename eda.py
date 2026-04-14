"""
Exploratory Data Analysis for the menstrual tracking dataset.

Loads `processed_period_data.csv` and produces a set of matplotlib figures
that help us understand feature distributions, missingness, correlations,
and cycle-phase behaviour — the foundation for the LSTM period-prediction
model.

Run:
    python eda.py

Outputs go to ./figures/*.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

CSV_PATH = HERE / "processed_period_data.csv"

LIKERT_MAP = {
    "Not at all": 0, "Very Low/Little": 1, "Very Low": 1,
    "Low": 2, "Moderate": 3, "High": 4, "Very High": 5,
}
PHASE_ORDER = ["Menstrual", "Follicular", "Fertility", "Luteal"]
PHASE_COLORS = {"Menstrual": "#e63946", "Follicular": "#f4a261",
                "Fertility": "#2a9d8f", "Luteal": "#264653"}

mpl.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- load & clean

LIKERT_COLS = ["appetite", "exerciselevel", "headaches", "cramps", "sorebreasts",
               "fatigue", "sleepissue", "moodswing", "stress", "foodcravings",
               "indigestion", "bloating"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # unify the two merge-artefact study_interval columns into one
    if "study_interval_x" in df.columns:
        df["study_interval"] = df["study_interval_x"]
        df = df.drop(columns=[c for c in ("study_interval_x", "study_interval_y") if c in df.columns])
    # force-convert every known Likert column; leaves already-numeric rows alone
    for c in LIKERT_COLS:
        if c in df.columns:
            s = df[c]
            if s.dtype.kind not in "fiu":  # not float/int/uint → map strings
                df[c] = s.astype(object).map(LIKERT_MAP).astype("float64")
    return df


# ---------------------------------------------------------------------- plots

def plot_missing(df: pd.DataFrame) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(miss.index, miss.values, color="#d62828")
    ax.invert_yaxis()
    ax.set_xlabel("Fraction missing")
    ax.set_title("Missing-data fraction per column")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIG_DIR / "01_missing_per_column.png")
    plt.close(fig)

    # cell-level missingness heatmap (sampled rows for readability)
    sample = df.sample(min(500, len(df)), random_state=0).sort_values(["id", "day_in_study"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(sample.isna().values, aspect="auto", cmap="Greys",
              interpolation="nearest")
    ax.set_xticks(range(len(sample.columns)))
    ax.set_xticklabels(sample.columns, rotation=90, fontsize=7)
    ax.set_ylabel("Row (sampled, sorted by id/day)")
    ax.set_title("Missing-value map (black = missing)")
    fig.savefig(FIG_DIR / "02_missing_heatmap.png")
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ("id", "day_in_study", "study_interval") if c in df.columns]
    )
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if not np.isnan(v) and abs(v) > 0.4:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.7 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Feature correlation heatmap")
    fig.savefig(FIG_DIR / "03_correlation_heatmap.png")
    plt.close(fig)


def plot_phase_distribution(df: pd.DataFrame) -> None:
    counts = df["phase"].value_counts().reindex(PHASE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values,
           color=[PHASE_COLORS[p] for p in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{int(v)}", ha="center", fontsize=9)
    ax.set_ylabel("Participant-days")
    ax.set_title("Cycle phase distribution across dataset")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "04_phase_distribution.png")
    plt.close(fig)


def plot_hormones_by_phase(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, col in zip(axes, ["lh", "estrogen", "pdg"]):
        if col not in df.columns:
            continue
        data = [df.loc[df["phase"] == p, col].dropna().values for p in PHASE_ORDER]
        bp = ax.boxplot(data, tick_labels=PHASE_ORDER, patch_artist=True,
                        showfliers=False)
        for patch, p in zip(bp["boxes"], PHASE_ORDER):
            patch.set_facecolor(PHASE_COLORS[p])
            patch.set_alpha(0.7)
        ax.set_title(f"{col.upper()} by phase")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Hormone levels across menstrual phases")
    fig.savefig(FIG_DIR / "05_hormones_by_phase.png")
    plt.close(fig)


def plot_symptom_by_phase_heatmap(df: pd.DataFrame) -> None:
    symptom_cols = ["headaches", "cramps", "sorebreasts", "fatigue",
                    "sleepissue", "moodswing", "stress", "foodcravings",
                    "indigestion", "bloating", "appetite", "exerciselevel"]
    symptom_cols = [c for c in symptom_cols if c in df.columns]
    mat = (df.groupby("phase")[symptom_cols].mean()
             .reindex(PHASE_ORDER))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(symptom_cols)))
    ax.set_xticklabels(symptom_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v > mat.values[~np.isnan(mat.values)].mean() else "black")
    fig.colorbar(im, ax=ax, label="Mean Likert (0-5)")
    ax.set_title("Average symptom intensity by cycle phase")
    fig.savefig(FIG_DIR / "06_symptom_by_phase.png")
    plt.close(fig)


def plot_hormone_trajectory(df: pd.DataFrame) -> None:
    # pick the participant with the most data, plot LH + estrogen over time
    pid = df.groupby("id").size().idxmax()
    sub = df[df["id"] == pid].sort_values("day_in_study")

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(sub["day_in_study"], sub["lh"], color="#e63946",
             marker="o", ms=3, lw=1, label="LH")
    ax2.plot(sub["day_in_study"], sub["estrogen"], color="#2a9d8f",
             marker="s", ms=3, lw=1, label="Estrogen")

    # shade phase bands
    for phase in PHASE_ORDER:
        mask = sub["phase"] == phase
        if mask.any():
            for _, row in sub[mask].iterrows():
                ax1.axvspan(row["day_in_study"] - 0.5, row["day_in_study"] + 0.5,
                            alpha=0.08, color=PHASE_COLORS[phase])

    ax1.set_xlabel("Day in study")
    ax1.set_ylabel("LH (mIU/mL)", color="#e63946")
    ax2.set_ylabel("Estrogen (ng/mL)", color="#2a9d8f")
    ax1.set_title(f"Hormone trajectory — participant {pid}")
    ax1.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "07_hormone_trajectory.png")
    plt.close(fig)


def plot_symptom_distributions(df: pd.DataFrame) -> None:
    cols = ["cramps", "fatigue", "sleepissue", "moodswing", "stress", "bloating"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.flat, cols):
        vals = df[col].dropna()
        ax.hist(vals, bins=np.arange(-0.5, 6.5, 1), color="#457b9d",
                edgecolor="white")
        ax.set_title(col)
        ax.set_xticks(range(6))
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Symptom intensity distributions (0 = none, 5 = very high)")
    fig.savefig(FIG_DIR / "08_symptom_distributions.png")
    plt.close(fig)


def plot_cycle_timeline(df: pd.DataFrame) -> None:
    """Show phase labels as colored bands across day_in_study for first 6 ids."""
    ids = sorted(df["id"].unique())[:6]
    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 1.2 * len(ids)),
                             sharex=True)
    if len(ids) == 1:
        axes = [axes]
    for ax, pid in zip(axes, ids):
        sub = df[df["id"] == pid].sort_values("day_in_study")
        for _, row in sub.iterrows():
            color = PHASE_COLORS.get(row["phase"], "#cccccc")
            ax.axvspan(row["day_in_study"] - 0.5, row["day_in_study"] + 0.5,
                       color=color, alpha=0.85)
        ax.set_yticks([])
        ax.set_ylabel(f"id={pid}", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Day in study")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[p], label=p)
               for p in PHASE_ORDER]
    axes[0].legend(handles=handles, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 1.8), frameon=False)
    fig.suptitle("Cycle-phase timelines (first 6 participants)", y=1.02)
    fig.savefig(FIG_DIR / "09_cycle_timelines.png")
    plt.close(fig)


def plot_period_start_intervals(df: pd.DataFrame) -> None:
    """Derive period_start from phase=Menstrual transitions and plot cycle-length histogram."""
    df = df.sort_values(["id", "study_interval", "day_in_study"]).copy()
    df["is_menstrual"] = (df["phase"] == "Menstrual").astype(int)

    starts = []
    for (pid, si), g in df.groupby(["id", "study_interval"]):
        last = -999
        for _, row in g.iterrows():
            if row["is_menstrual"] == 1 and row["day_in_study"] - last > 7:
                starts.append({"id": pid, "si": si, "day": row["day_in_study"]})
            if row["is_menstrual"] == 1:
                last = row["day_in_study"]

    sdf = pd.DataFrame(starts)
    sdf["next_day"] = sdf.groupby(["id", "si"])["day"].shift(-1)
    sdf["cycle_length"] = sdf["next_day"] - sdf["day"]
    cl = sdf["cycle_length"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cl, bins=np.arange(15, 55, 2), color="#e63946", edgecolor="white")
    ax.axvline(cl.mean(), color="black", ls="--",
               label=f"mean {cl.mean():.1f} d")
    ax.axvline(cl.median(), color="black", ls=":",
               label=f"median {cl.median():.0f} d")
    ax.set_xlabel("Cycle length (days)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cycle-length distribution (n={len(cl)} cycles)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "10_cycle_length_histogram.png")
    plt.close(fig)


# ---------------------------------------------------------------------- main

def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols from "
          f"{CSV_PATH.name}")
    print(f"  participants: {df['id'].nunique()}  "
          f"| intervals: {sorted(df['study_interval'].unique())}")

    plot_missing(df)
    plot_correlation(df)
    plot_phase_distribution(df)
    plot_hormones_by_phase(df)
    plot_symptom_by_phase_heatmap(df)
    plot_hormone_trajectory(df)
    plot_symptom_distributions(df)
    plot_cycle_timeline(df)
    plot_period_start_intervals(df)

    print(f"\n[ok] Wrote {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
