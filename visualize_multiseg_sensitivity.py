import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RHO = 0.25

SEGMENT_LST = [
    '3420 - Auto Mall NB',
    '3430 - Mowry NB',
    '3440 - Decoto/84 NB',
    '3450 - Whipple NB',
    '3460 - Hesperian/238 NB'
]

TOLL_COLS = [f"Toll {i}" for i in range(5)]
OBJECTIVE_COLS = [
    "Total Travel Time",
    "Total Emission",
    "Total Utility Cost",
    "Total Revenue",
]

OBJECTIVE_DIRECTIONS = {
    "Total Travel Time": "min",
    "Total Emission": "min",
    "Total Utility Cost": "min",
    "Total Revenue": "max",
}

SAVE_DIR = "DynamicDesign/MultiSeg/Sensitivity"

df_design = pd.read_csv("./toll_design_multiseg_hour=16_multi-rho.csv")
df_design = df_design[df_design["Rho"] == RHO].copy()

df_design = df_design[(df_design["Toll 0"] > 0) & (df_design["Toll 1"] > 0) & (df_design["Toll 2"] > 0) & (df_design["Toll 3"] > 0) & (df_design["Toll 4"] > 0)].copy()

df_toll = pd.read_csv("data/df_toll.csv")

os.makedirs(SAVE_DIR, exist_ok=True)


def _sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _sanitize_hour(hour) -> str:
    return _sanitize_filename(str(hour))


def _round_to_half(x):
    return np.round(x * 2) / 2


def compute_current_toll_vector_by_hour(df_toll, segment_lst):
    """
    Compute average toll per segment for each hour across all dates,
    then round to the nearest 0.5.

    Returns
    -------
    dict:
        hour -> np.array of shape (len(segment_lst),)
    """
    grouped = (
        df_toll
        .groupby(["Hour", "Segment"])["Avg_total_toll"]
        .mean()
        .reset_index()
    )

    toll_by_hour = {}

    for hour, sub in grouped.groupby("Hour"):
        toll_map = dict(zip(sub["Segment"], sub["Avg_total_toll"]))
        current_tolls = []
        for seg in segment_lst:
            if seg not in toll_map:
                raise ValueError(f"Segment {seg} not found in df_toll for hour {hour}")
            val = toll_map[seg]
            current_tolls.append(min(_round_to_half(val), 5.0))
        toll_by_hour[hour] = np.array(current_tolls, dtype=float)

    return toll_by_hour


def _get_best_row(df: pd.DataFrame, objective_col: str, direction: str):
    if direction == "min":
        idx = df[objective_col].idxmin()
    elif direction == "max":
        idx = df[objective_col].idxmax()
    else:
        raise ValueError(f"direction must be 'min' or 'max', got {direction}")
    return df.loc[idx]


def _get_near_optimal_df(
    df: pd.DataFrame,
    objective_col: str,
    direction: str,
    rel_tol: float = 0.01,
):
    best = df[objective_col].min() if direction == "min" else df[objective_col].max()
    scale = max(abs(best), 1e-12)

    if direction == "min":
        mask = df[objective_col] <= best + rel_tol * scale
    else:
        mask = df[objective_col] >= best - rel_tol * scale

    return df.loc[mask].copy(), best


def plot_1d_sensitivity_curves(
    df: pd.DataFrame,
    hour,
    objective_cols=OBJECTIVE_COLS,
    toll_cols=TOLL_COLS,
    segment_names=SEGMENT_LST,
    directions=OBJECTIVE_DIRECTIONS,
    save_dir: str = SAVE_DIR,
    fix_at: str = "current",
    current_toll_vector=None,
    add_best_line: bool = True,
):
    """
    For each objective and each segment, generate ONE plot per segment
    (instead of putting all segments into a single figure).
    """
    hour_dir = os.path.join(save_dir, f"Hour_{_sanitize_hour(hour)}")
    os.makedirs(hour_dir, exist_ok=True)

    if fix_at == "current":
        label = "Current Toll"
    else:
        label = "Optimal Toll"

    if df.empty:
        print(f"[Warning] Empty design dataframe for hour={hour}. Skipping sensitivity plots.")
        return

    for objective_col in objective_cols:
        direction = directions[objective_col]

        if fix_at == "optimal":
            ref_row = _get_best_row(df, objective_col, direction)
            ref_tolls = ref_row[toll_cols].to_numpy(dtype=float)
            ref_label = "objective-specific optimum"
        elif fix_at == "current":
            if current_toll_vector is None:
                raise ValueError("current_toll_vector must be provided when fix_at='current'")
            ref_tolls = np.asarray(current_toll_vector, dtype=float)
            if len(ref_tolls) != len(toll_cols):
                raise ValueError("current_toll_vector must have length 5")
            ref_label = "current toll vector"
        else:
            raise ValueError("fix_at must be either 'optimal' or 'current'")

        best_val = df[objective_col].min() if direction == "min" else df[objective_col].max()

        # 🔥 Key change: loop over segments and create ONE figure per segment
        for j, (toll_col, seg_name) in enumerate(zip(toll_cols, segment_names)):

            fig, ax = plt.subplots(figsize=(5, 4))  # one plot per segment

            mask = np.ones(len(df), dtype=bool)
            for k, other_col in enumerate(toll_cols):
                if k == j:
                    continue
                mask &= np.isclose(df[other_col].to_numpy(dtype=float), ref_tolls[k])

            sub = df.loc[mask, [toll_col, objective_col]].copy()
            sub = sub.sort_values(toll_col)

            if sub.empty:
                ax.text(
                    0.5, 0.5,
                    "No matching rows\nfor this slice",
                    ha="center", va="center", transform=ax.transAxes
                )
                ax.set_title(f"{seg_name}\n({fix_at} = {ref_tolls[j]:.1f})")
                ax.set_xlabel("Toll")
                ax.set_ylabel(objective_col)
            else:
                ax.plot(sub[toll_col], sub[objective_col], marker="o")

#                if add_best_line:
#                    ax.axhline(best_val, linestyle="--", linewidth=1)

                ax.axvline(ref_tolls[j], color="red", label=label)
                ax.legend()

#                ax.set_title(f"{seg_name}\n({fix_at} = {ref_tolls[j]:.1f})")
#                ax.set_xlabel("Toll")
                ax.set_ylabel(objective_col)
                ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.2f}"))
                ax.grid(alpha=0.25)

#            fig.suptitle(
#                f"Hour {hour}: {objective_col}\n(other 4 tolls fixed at {ref_label})"
#            )

            fig.tight_layout()

            fname = (
                f"sensitivity_hour_{_sanitize_hour(hour)}_"
                f"{_sanitize_filename(objective_col)}_"
                f"{_sanitize_filename(seg_name)}_"
                f"{fix_at}_rho={RHO}.png"
            )

            fig.savefig(os.path.join(hour_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_near_optimal_regions(
    df: pd.DataFrame,
    hour,
    objective_cols=OBJECTIVE_COLS,
    toll_cols=TOLL_COLS,
    segment_names=SEGMENT_LST,
    directions=OBJECTIVE_DIRECTIONS,
    rel_tol: float = 0.01,
    save_dir: str = SAVE_DIR,
    show_points: bool = True,
):
    """
    For each objective, collect all toll vectors within rel_tol of the optimum,
    then show the near-optimal range for each segment.
    """
    hour_dir = os.path.join(save_dir, f"Hour_{_sanitize_hour(hour)}")
    os.makedirs(hour_dir, exist_ok=True)

    if df.empty:
        print(f"[Warning] Empty design dataframe for hour={hour}. Skipping near-optimal plots.")
        return

    for objective_col in objective_cols:
        direction = directions[objective_col]
        near_df, best_val = _get_near_optimal_df(
            df, objective_col, direction=direction, rel_tol=rel_tol
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        data = [near_df[c].to_numpy(dtype=float) for c in toll_cols]
        positions = np.arange(1, len(toll_cols) + 1)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=False,
            showfliers=False
        )

        if show_points:
            rng = np.random.default_rng(12345)
            for i, vals in enumerate(data, start=1):
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter,
                    vals,
                    s=18,
                    alpha=0.5
                )

        mins = [np.min(vals) if len(vals) > 0 else np.nan for vals in data]
        maxs = [np.max(vals) if len(vals) > 0 else np.nan for vals in data]

        for i, (mn, mx) in enumerate(zip(mins, maxs), start=1):
            if np.isfinite(mn):
                ax.text(i, mx + 0.05, f"[{mn:.1f}, {mx:.1f}]", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(positions)
        ax.set_xticklabels(segment_names, rotation=20, ha="right")
        ax.set_ylabel("Toll")
        ax.set_ylim(
            bottom=min(df[toll_cols].min()) - 0.1,
            top=max(df[toll_cols].max()) + 0.4
        )
        ax.grid(axis="y", alpha=0.25)

        n_near = len(near_df)
        pct_near = 100 * n_near / max(len(df), 1)
        tol_pct = 100 * rel_tol

        ax.set_title(
            f"Hour {hour}: Near-optimal toll regions for {objective_col}\n"
            f"({direction}-objective, within {tol_pct:.1f}% of optimum; "
            f"{n_near} solutions = {pct_near:.2f}% of grid)"
        )

        fig.tight_layout()
        fname = (
            f"near_optimal_hour_{_sanitize_hour(hour)}_"
            f"{_sanitize_filename(objective_col)}_{int(round(tol_pct * 10))}bp.png"
        )
        fig.savefig(os.path.join(hour_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_near_optimal_summary_table(
    df: pd.DataFrame,
    hour,
    objective_cols=OBJECTIVE_COLS,
    toll_cols=TOLL_COLS,
    segment_names=SEGMENT_LST,
    directions=OBJECTIVE_DIRECTIONS,
    rel_tol: float = 0.01,
    save_dir: str = SAVE_DIR,
):
    """
    Save a CSV summary with min / median / max toll among near-optimal solutions.
    """
    hour_dir = os.path.join(save_dir, f"Hour_{_sanitize_hour(hour)}")
    os.makedirs(hour_dir, exist_ok=True)

    rows = []

    if df.empty:
        print(f"[Warning] Empty design dataframe for hour={hour}. Skipping summary table.")
        return pd.DataFrame()

    for objective_col in objective_cols:
        direction = directions[objective_col]
        near_df, best_val = _get_near_optimal_df(df, objective_col, direction, rel_tol=rel_tol)

        for toll_col, seg_name in zip(toll_cols, segment_names):
            vals = near_df[toll_col].to_numpy(dtype=float)
            rows.append({
                "Hour": hour,
                "Objective": objective_col,
                "Direction": direction,
                "Segment": seg_name,
                "TollColumn": toll_col,
                "BestObjectiveValue": best_val,
                "NumNearOptimalSolutions": len(near_df),
                "MinNearOptimalToll": np.min(vals) if len(vals) else np.nan,
                "MedianNearOptimalToll": np.median(vals) if len(vals) else np.nan,
                "MaxNearOptimalToll": np.max(vals) if len(vals) else np.nan,
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(hour_dir, f"near_optimal_summary_hour_{_sanitize_hour(hour)}.csv"), index=False)
    return out


def run_all_hourly_plots(
    df_design,
    df_toll,
    save_dir=SAVE_DIR,
    rel_tol=0.01,
    fix_at="current",
):
    """
    Generate all plots separately for each hour appearing in df_design.
    """
    os.makedirs(save_dir, exist_ok=True)

    toll_by_hour = compute_current_toll_vector_by_hour(df_toll, SEGMENT_LST)

    design_hours = sorted(df_design["Hour"].dropna().unique().tolist())
    summary_lst = []

    for hour in design_hours:
        df_hour = df_design[df_design["Hour"] == hour].copy()

        if df_hour.empty:
            print(f"[Info] No design rows for hour={hour}. Skipping.")
            continue

        if fix_at == "current":
            if hour not in toll_by_hour:
                print(f"[Warning] No current toll data for hour={hour}. Skipping sensitivity plots.")
                current_toll_vector = None
            else:
                current_toll_vector = toll_by_hour[hour]
        else:
            current_toll_vector = None

        print(f"Processing hour={hour}...")

        plot_1d_sensitivity_curves(
            df=df_hour,
            hour=hour,
            objective_cols=OBJECTIVE_COLS,
            toll_cols=TOLL_COLS,
            segment_names=SEGMENT_LST,
            directions=OBJECTIVE_DIRECTIONS,
            save_dir=save_dir,
            fix_at=fix_at,
            current_toll_vector=current_toll_vector,
            add_best_line=True,
        )
        
#        plot_1d_sensitivity_curves(
#            df=df_hour,
#            hour=hour,
#            objective_cols=OBJECTIVE_COLS,
#            toll_cols=TOLL_COLS,
#            segment_names=SEGMENT_LST,
#            directions=OBJECTIVE_DIRECTIONS,
#            save_dir=save_dir,
#            fix_at="optimal",
#            add_best_line=True,
#        )

#        plot_near_optimal_regions(
#            df=df_hour,
#            hour=hour,
#            objective_cols=OBJECTIVE_COLS,
#            toll_cols=TOLL_COLS,
#            segment_names=SEGMENT_LST,
#            directions=OBJECTIVE_DIRECTIONS,
#            rel_tol=rel_tol,
#            save_dir=save_dir,
#            show_points=True,
#        )
#
#        summary_df = make_near_optimal_summary_table(
#            df=df_hour,
#            hour=hour,
#            objective_cols=OBJECTIVE_COLS,
#            toll_cols=TOLL_COLS,
#            segment_names=SEGMENT_LST,
#            directions=OBJECTIVE_DIRECTIONS,
#            rel_tol=rel_tol,
#            save_dir=save_dir,
#        )
#        if not summary_df.empty:
#            summary_lst.append(summary_df)

    if summary_lst:
        summary_all = pd.concat(summary_lst, ignore_index=True)
        summary_all.to_csv(os.path.join(save_dir, "near_optimal_summary_all_hours.csv"), index=False)
        return summary_all

    return pd.DataFrame()


# =========================
# Run everything hour by hour
# =========================
summary_df = run_all_hourly_plots(
    df_design=df_design,
    df_toll=df_toll,
    save_dir=SAVE_DIR,
    rel_tol=0.01,
    fix_at="current",
)

#print(summary_df.head())
