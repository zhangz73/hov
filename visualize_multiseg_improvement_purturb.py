import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

os.makedirs("DynamicDesign/MultiSeg/ImprovementsPerturbCombined", exist_ok=True)

PERTURB_X_LST = [0,1,2,3,4,5] #[-4, -2, -1, 1, 2, 4]
SEGMENT_LST = [
    '3420 - Auto Mall NB',
    '3430 - Mowry NB',
    '3440 - Decoto/84 NB',
    '3450 - Whipple NB',
    '3460 - Hesperian/238 NB'
]

df_design = pd.read_csv("./toll_design_multiseg.csv")
df_design = df_design[df_design["Rho"] == 0.25].copy()
df_toll = pd.read_csv("data/df_toll.csv")
N_HOURS = 12

N_POP = 1
df_design["Total Travel Time"] /= N_POP
df_design["Total Emission"] /= N_POP
df_design["Total Utility Cost"] /= N_POP


def round_to_half(x):
    if np.isscalar(x):
        return np.round(x * 2) / 2
    arr = np.asarray(x, dtype=float)
    return np.round(arr * 2) / 2


def compute_hourly_actual_toll_vector(df_toll, hour):
    """
    Compute the average actual toll vector across all 5 segments at a given hour,
    rounded to the nearest 0.5.
    """
    toll_vec = []
    for seg in SEGMENT_LST:
        df_seg = df_toll[(df_toll["Hour"] == hour) & (df_toll["Segment"] == seg)]
        if df_seg.empty:
            raise ValueError(f"No toll data found for hour={hour}, segment={seg}")
        toll_avg = df_seg["Avg_total_toll"].mean()
        toll_vec.append(round_to_half(toll_avg))
    return np.array(toll_vec, dtype=float)


def get_current_row_from_vector(df_design_curr, toll_vec):
    """
    Return the row in df_design_curr that exactly matches toll_vec.
    If no exact match exists, return the nearest row in Euclidean distance.
    """
    exact_mask = np.ones(len(df_design_curr), dtype=bool)
    for i in range(len(toll_vec)):
        exact_mask &= np.isclose(df_design_curr[f"Toll {i}"].to_numpy(dtype=float), toll_vec[i])

    if exact_mask.sum() > 0:
        return df_design_curr.loc[exact_mask].iloc[0]

    X = df_design_curr[[f"Toll {i}" for i in range(len(toll_vec))]].to_numpy(dtype=float)
    dist = ((X - toll_vec[None, :]) ** 2).sum(axis=1)
    return df_design_curr.iloc[np.argmin(dist)]


def get_slice_with_segment_perturbation(df_design_curr, base_toll_vec, segment_idx, x_max):
    """
    Keep all other segment tolls fixed at base_toll_vec, and allow the selected segment
    to vary within the interval between base_toll and base_toll + x_max, clipped to [0, 5].
    """
    lower = max(0.0, min(5.0, min(base_toll_vec[segment_idx], base_toll_vec[segment_idx] - x_max)))
    upper = max(0.0, min(5.0, max(base_toll_vec[segment_idx], base_toll_vec[segment_idx] + x_max)))
    base_toll_vec = round_to_half(base_toll_vec)
    base_toll_vec = [max(min(5.0, x), 0.0) for x in base_toll_vec]

    mask = np.ones(len(df_design_curr), dtype=bool)
    for j in range(len(base_toll_vec)):
        col = f"Toll {j}"
        vals = df_design_curr[col].to_numpy(dtype=float)
        if j == segment_idx:
            mask &= (vals >= lower - 1e-9) & (vals <= upper + 1e-9)
        else:
            mask &= np.isclose(vals, base_toll_vec[j])

    return df_design_curr.loc[mask].copy()


def compute_improvement_from_slice(df_slice, curr_row, objective_col, goal_type):
    """
    goal_type in {"min", "max"}.
    Returns:
        pct_improvement, nominal_improvement, best_row
    """
    curr_val = curr_row[objective_col]

    if df_slice.empty:
        return np.nan, np.nan, None

    if goal_type == "min":
        best_idx = df_slice[objective_col].idxmin()
        best_val = df_slice.loc[best_idx, objective_col]
        nominal = curr_val - best_val
        pct = (nominal / curr_val * 100) if curr_val != 0 else np.nan
    elif goal_type == "max":
        best_idx = df_slice[objective_col].idxmax()
        best_val = df_slice.loc[best_idx, objective_col]
        nominal = best_val - curr_val
        pct = (nominal / curr_val * 100) if curr_val != 0 else np.nan
    else:
        raise ValueError(f"Unknown goal_type={goal_type}")

    return pct, nominal, df_slice.loc[best_idx]


def _compute_global_ylim(all_plot_data, metric_key, value_type, pad_frac=0.05):
    vals = []
    for hour in all_plot_data[metric_key]:
        for segment_short in all_plot_data[metric_key][hour]:
            arr = np.asarray(all_plot_data[metric_key][hour][segment_short][value_type], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)

    if len(vals) == 0:
        return None

    vals = np.concatenate(vals)
    ymin = np.min(vals)
    ymax = np.max(vals)

    if np.isclose(ymin, ymax):
        delta = 1.0 if np.isclose(ymin, 0.0) else 0.05 * abs(ymin)
        return ymin - delta, ymax + delta

    pad = pad_frac * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_hour_objective_all_segments(
    x_lst,
    segment_pct_dct,
    segment_value_dct,
    goal,
    hour,
    pct_ylim=None,
    value_ylim=None,
):
    """
    One plot for a given hour and objective, containing all 5 segments.
    Left y-axis: percentage improvement
    Right y-axis: nominal improvement
    """
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax2 = ax1.twinx()

    handles = []
    labels = []

    for segment_short in segment_pct_dct:
        line1, = ax1.plot(
            x_lst,
            segment_pct_dct[segment_short],
            marker="o",
            linestyle="-",
            alpha=0.9,
            label=f"{segment_short} (pct)"
        )
        ax2.plot(
            x_lst,
            segment_value_dct[segment_short],
            marker="x",
            linestyle="--",
            alpha=0.9,
            color=line1.get_color(),
            label=f"{segment_short} (nominal)"
        )
        handles.append(line1)
        labels.append(segment_short)

    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1)

    ax1.set_xlabel("Perturbation ($)")
    ax1.set_ylabel("Pct. Improvement")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    if pct_ylim is not None:
        ax1.set_ylim(*pct_ylim)

    ax2.set_ylabel("Nominal Improvement")
    if goal in ["Max Revenue", "Min Utility Cost"]:
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    else:
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f} mins"))
    if value_ylim is not None:
        ax2.set_ylim(*value_ylim)

    ax1.grid(alpha=0.3)
    ax1.legend(handles, labels, loc="upper left", title="Segment")

    plt.title(f"Hour {hour}, {goal}")
    plt.tight_layout()
    plt.savefig(
        f"DynamicDesign/MultiSeg/ImprovementsPerturbCombined/"
        f"hour_{hour}_{goal.lower().replace(' ', '_')}_all_segments.png"
    )
    plt.clf()
    plt.close()


metric_specs = {
    "Min Congestion": ("Total Travel Time", "min"),
    "Min Emission": ("Total Emission", "min"),
    "Max Revenue": ("Total Revenue", "max"),
    "Min Utility Cost": ("Total Utility Cost", "min"),
}

segment_short_map = {
    seg: seg.split("-")[1].split("/")[0].strip()
    for seg in SEGMENT_LST
}

# =========================================================
# First pass: compute all improvements
# all_plot_data[metric][hour][segment_short] = {"x": ..., "pct": ..., "value": ...}
# =========================================================

all_plot_data = {
    metric_name: {} for metric_name in metric_specs
}

for hour_idx in range(N_HOURS):
    hour = 7 + hour_idx
    df_design_curr = df_design[df_design["Hour"] == hour].copy()
    base_toll_vec = compute_hourly_actual_toll_vector(df_toll, hour)
    curr_row = get_current_row_from_vector(df_design_curr, base_toll_vec)

    for metric_name, (objective_col, goal_type) in metric_specs.items():
        all_plot_data[metric_name][hour] = {}

        for segment_idx, segment in enumerate(SEGMENT_LST):
            segment_short = segment_short_map[segment]

            pct_lst = []
            value_lst = []

            for x in PERTURB_X_LST:
                df_slice = get_slice_with_segment_perturbation(
                    df_design_curr=df_design_curr,
                    base_toll_vec=base_toll_vec,
                    segment_idx=segment_idx,
                    x_max=x,
                )
                pct, nominal, _ = compute_improvement_from_slice(
                    df_slice, curr_row, objective_col, goal_type
                )
                pct_lst.append(pct)
                value_lst.append(nominal)

            all_plot_data[metric_name][hour][segment_short] = {
                "x": list(PERTURB_X_LST),
                "pct": pct_lst,
                "value": value_lst,
            }


# =========================================================
# Second pass: compute shared y-limits across all hours for each objective
# =========================================================

shared_ylims = {}
for metric_name in metric_specs:
    shared_ylims[metric_name] = {
        "pct": _compute_global_ylim(all_plot_data, metric_name, "pct"),
        "value": _compute_global_ylim(all_plot_data, metric_name, "value"),
    }


# =========================================================
# Third pass: make one plot for each hour and each objective
# =========================================================

for metric_name in metric_specs:
    for hour in sorted(all_plot_data[metric_name].keys()):
        segment_pct_dct = {}
        segment_value_dct = {}

        for segment_short in all_plot_data[metric_name][hour]:
            segment_pct_dct[segment_short] = all_plot_data[metric_name][hour][segment_short]["pct"]
            segment_value_dct[segment_short] = all_plot_data[metric_name][hour][segment_short]["value"]

        plot_hour_objective_all_segments(
            x_lst=PERTURB_X_LST,
            segment_pct_dct=segment_pct_dct,
            segment_value_dct=segment_value_dct,
            goal=metric_name,
            hour=hour,
            pct_ylim=shared_ylims[metric_name]["pct"],
            value_ylim=shared_ylims[metric_name]["value"],
        )
