import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =========================================================
# Config
# =========================================================

SAVE_DIR = "DynamicDesign/MultiSeg/ImprovementsJoint"
os.makedirs(SAVE_DIR, exist_ok=True)

BUDGET_LST = [1, 2, 4, 8, 16] #[1, 2, 3, 4, 5]
SEGMENT_LST = [
    "3420 - Auto Mall NB",
    "3430 - Mowry NB",
    "3440 - Decoto/84 NB",
    "3450 - Whipple NB",
    "3460 - Hesperian/238 NB",
]
TOLL_COLS = [f"Toll {i}" for i in range(5)]

OBJECTIVE_SPECS = {
    "Min Congestion": ("Total Travel Time", "min"),
    "Min Emission": ("Total Emission", "min"),
    "Max Revenue": ("Total Revenue", "max"),
    "Min Utility Cost": ("Total Utility Cost", "min"),
}

N_HOURS = 12
START_HOUR = 7
N_POP = 1

# =========================================================
# Load data
# =========================================================

df_design = pd.read_csv("./toll_design_multiseg.csv")
df_design = df_design[df_design["Rho"] == 0.25].copy()

df_toll = pd.read_csv("data/df_toll.csv").copy()

# Optional normalization to match the user's earlier convention
for col in ["Total Travel Time", "Total Emission", "Total Utility Cost"]:
    if col in df_design.columns:
        df_design[col] = df_design[col] / N_POP


# =========================================================
# Helpers
# =========================================================

def round_to_half(x):
    arr = np.asarray(x, dtype=float)
    return np.round(arr * 2) / 2


def compute_hourly_actual_toll_vector(df_toll, hour):
    """
    Average actual toll across all dates for each segment at a given hour,
    rounded to the nearest 0.5.
    """
    toll_vec = []
    for seg in SEGMENT_LST:
        df_seg = df_toll[(df_toll["Hour"] == hour) & (df_toll["Segment"] == seg)]
        if df_seg.empty:
            raise ValueError(f"No toll data found for hour={hour}, segment={seg}")
        toll_avg = df_seg["Avg_total_toll"].mean()
        toll_vec.append(toll_avg)
    return round_to_half(toll_vec)


def get_current_row_from_vector(df_design_curr, toll_vec):
    """
    Return the row in df_design_curr that exactly matches toll_vec.
    If no exact match exists, return the nearest row in Euclidean distance.
    """
    toll_vec = [min(5.0, max(0.0, round_to_half(x))) for x in toll_vec]
    toll_vec = np.asarray(toll_vec, dtype=float)

    exact_mask = np.ones(len(df_design_curr), dtype=bool)
    for i, col in enumerate(TOLL_COLS):
        exact_mask &= np.isclose(df_design_curr[col].to_numpy(dtype=float), toll_vec[i])

    if exact_mask.sum() > 0:
        return df_design_curr.loc[exact_mask].iloc[0]
    
    X = df_design_curr[TOLL_COLS].to_numpy(dtype=float)
    dist = ((X - toll_vec[None, :]) ** 2).sum(axis=1)
    return df_design_curr.iloc[np.argmin(dist)]


def get_joint_budget_slice(df_design_curr, base_toll_vec, budget):
    """
    Keep all design rows whose total absolute perturbation from the base toll vector
    is at most `budget`, i.e.
        sum_s |tau_s - tau_s^base| <= budget.
    """
    X = df_design_curr[TOLL_COLS].to_numpy(dtype=float)
    base_toll_vec = np.asarray(base_toll_vec, dtype=float)
    l1_dist = np.abs(X - base_toll_vec[None, :]).sum(axis=1)
    return df_design_curr.loc[l1_dist <= budget + 1e-9].copy()


def compute_improvement_from_slice(df_slice, curr_row, objective_col, goal_type):
    """
    Returns:
        pct_improvement, nominal_improvement, best_row
    """
    curr_val = float(curr_row[objective_col])

    if df_slice.empty:
        return np.nan, np.nan, None

    if goal_type == "min":
        best_idx = df_slice[objective_col].idxmin()
        best_val = float(df_slice.loc[best_idx, objective_col])
        nominal = curr_val - best_val
        pct = (nominal / curr_val * 100.0) if curr_val != 0 else np.nan
    elif goal_type == "max":
        best_idx = df_slice[objective_col].idxmax()
        best_val = float(df_slice.loc[best_idx, objective_col])
        nominal = best_val - curr_val
        pct = (nominal / curr_val * 100.0) if curr_val != 0 else np.nan
    else:
        raise ValueError(f"Unknown goal_type={goal_type}")

    return pct, nominal, df_slice.loc[best_idx]


def _compute_shared_ylim(series_dict, pad_frac=0.05):
    """
    series_dict[budget] = list over hours
    """
    vals = []
    for budget in series_dict:
        arr = np.asarray(series_dict[budget], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.append(arr)

    if not vals:
        return None

    vals = np.concatenate(vals)
    ymin = np.min(vals)
    ymax = np.max(vals)

    if np.isclose(ymin, ymax):
        delta = 1.0 if np.isclose(ymin, 0.0) else 0.05 * abs(ymin)
        return ymin - delta, ymax + delta

    pad = pad_frac * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_joint_improvement(hour_lst, series_dict, goal, value_type, ylim=None):
    """
    value_type in {"pct", "nominal"}
    Produces one plot per objective and value type.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for budget in BUDGET_LST:
        ax.plot(
            hour_lst,
            series_dict[budget],
            marker="o",
            label=f"Total perturbation up to ${budget}",
        )

    ax.set_xlabel("Hour")

    if value_type == "pct":
        ax.set_ylabel("Pct. Improvement")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        fname = f"{goal.lower().replace(' ', '_')}_pct.png"
    else:
        ax.set_ylabel("Nominal Improvement")
        if goal in ["Max Revenue", "Min Utility Cost"]:
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        else:
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f} mins"))
        fname = f"{goal.lower().replace(' ', '_')}_nominal.png"

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(goal)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main computation
# =========================================================

all_results = {}

for goal, (objective_col, goal_type) in OBJECTIVE_SPECS.items():
    pct_series = {b: [] for b in BUDGET_LST}
    nominal_series = {b: [] for b in BUDGET_LST}
    hour_lst = []

    for hour_idx in range(N_HOURS):
        hour = START_HOUR + hour_idx
        hour_lst.append(hour)

        df_design_curr = df_design[df_design["Hour"] == hour].copy()
        if df_design_curr.empty:
            for b in BUDGET_LST:
                pct_series[b].append(np.nan)
                nominal_series[b].append(np.nan)
            continue

        base_toll_vec = compute_hourly_actual_toll_vector(df_toll, hour)
        curr_row = get_current_row_from_vector(df_design_curr, base_toll_vec)

        for budget in BUDGET_LST:
            df_slice = get_joint_budget_slice(df_design_curr, base_toll_vec, budget)
            pct, nominal, _ = compute_improvement_from_slice(
                df_slice=df_slice,
                curr_row=curr_row,
                objective_col=objective_col,
                goal_type=goal_type,
            )
            pct_series[budget].append(pct)
            nominal_series[budget].append(nominal)

    all_results[goal] = {
        "hour_lst": hour_lst,
        "pct": pct_series,
        "nominal": nominal_series,
    }

# Shared y-ranges across budgets within each objective/value type
for goal in OBJECTIVE_SPECS:
    pct_ylim = _compute_shared_ylim(all_results[goal]["pct"])
    nominal_ylim = _compute_shared_ylim(all_results[goal]["nominal"])

    plot_joint_improvement(
        hour_lst=all_results[goal]["hour_lst"],
        series_dict=all_results[goal]["pct"],
        goal=goal,
        value_type="pct",
        ylim=pct_ylim,
    )
    plot_joint_improvement(
        hour_lst=all_results[goal]["hour_lst"],
        series_dict=all_results[goal]["nominal"],
        goal=goal,
        value_type="nominal",
        ylim=nominal_ylim,
    )

## Also save the underlying data
#rows = []
#for goal in OBJECTIVE_SPECS:
#    for value_type in ["pct", "nominal"]:
#        for budget in BUDGET_LST:
#            for hour, value in zip(all_results[goal]["hour_lst"], all_results[goal][value_type][budget]):
#                rows.append({
#                    "Objective": goal,
#                    "ValueType": value_type,
#                    "Hour": hour,
#                    "Budget": budget,
#                    "Value": value,
#                })
#
#out_df = pd.DataFrame(rows)
#out_df.to_csv(os.path.join(SAVE_DIR, "joint_perturbation_improvements.csv"), index=False)
#
#print(f"Saved plots and CSV to: {SAVE_DIR}")
