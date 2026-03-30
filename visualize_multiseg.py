import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

os.makedirs("DynamicDesign/MultiSeg/Improvements", exist_ok=True)
PLOT_IMPROVEMENT = False

TOLL_COLS = [f"Toll {i}" for i in range(5)]

SEGMENT_LST = ['3420 - Auto Mall NB', '3430 - Mowry NB', '3440 - Decoto/84 NB', '3450 - Whipple NB', '3460 - Hesperian/238 NB']
df_design = pd.read_csv("./toll_design_multiseg.csv")
df_design = df_design[df_design["Rho"] == 0.25]
df_design = df_design[(df_design["Toll 0"] > 0) & (df_design["Toll 1"] > 0) & (df_design["Toll 2"] > 0) & (df_design["Toll 3"] > 0) & (df_design["Toll 4"] > 0)]
df_design = df_design[df_design["Loss"] <= 1e-4]
## Date, Hour, Segment, Avg_total_toll
df_toll = pd.read_csv("data/df_toll.csv")
N_HOURS = 12

INT_GRID = 10
N_POP = 1#INT_GRID ** 3 # 24546
df_design["Total Travel Time"] /= N_POP
df_design["Total Emission"] /= N_POP
df_design["Total Utility Cost"] /= N_POP

toll_cols = [f"Toll {i}" for i in range(3)]
mask = (df_design["Hour"] <= 13) & (df_design[toll_cols].ge(1.5).any(axis=1))
df_design = df_design.loc[~mask].reset_index(drop=True)
toll_cols = [f"Toll {i}" for i in range(5)]
mask = (df_design["Hour"] <= 13) & (df_design[toll_cols].ge(2.5).any(axis=1))
df_design = df_design.loc[~mask].reset_index(drop=True)

if not PLOT_IMPROVEMENT:
    TOLL_LAM = 2e-1
    df_design_lst = []
    for hour_idx in range(N_HOURS):
        hour = 7 + hour_idx
        toll_deviate = 0
        df_design_curr = df_design[df_design["Hour"] == hour].copy()
        for i in range(len(SEGMENT_LST)):
            segment = SEGMENT_LST[i]
            df_toll_curr = df_toll[(df_toll["Hour"] == hour) & (df_toll["Segment"] == segment)]
            toll_avg = df_toll_curr["Avg_total_toll"].mean()
            toll_deviate += (toll_avg - df_design_curr[f"Toll {i}"]).abs()
        df_design_curr["Total Travel Time"] += toll_deviate * df_design_curr["Total Travel Time"].std() * TOLL_LAM
        df_design_curr["Total Emission"] += toll_deviate * df_design_curr["Total Emission"].std() * TOLL_LAM
        df_design_curr["Total Utility Cost"] += toll_deviate * df_design_curr["Total Utility Cost"].std() * TOLL_LAM
        df_design_curr["Total Revenue"] -= toll_deviate * df_design_curr["Total Revenue"].std() * TOLL_LAM
        df_design_lst.append(df_design_curr)
    df_design = pd.concat(df_design_lst, ignore_index = True)

def plot_hourly_price(hour_lst, toll_design_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, goal, segment):
    if goal is not None:
        plt.plot(hour_lst, toll_design_lst, color = "red", label = "Optimal Toll Price")
    plt.scatter(hour_lst, toll_avg_lst, color = "blue", label = "Average Actual Tolls")
    plt.fill_between(hour_lst, toll_lower_lst, toll_upper_lst, color = "blue", alpha = 0.2, label = "95% CI of Actual Tolls")
#    plt.gcf().axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
#    plt.gcf().autofmt_xdate()
    plt.xlabel("Time of Day")
    if goal is not None:
        plt.ylabel(f"{segment} - {goal.replace('Utility ', '')}")
    plt.legend(loc = "upper left")
    if goal is not None:
        plt.savefig(f"DynamicDesign/MultiSeg/{segment.lower().replace(' ', '_')}_{goal.lower().replace(' ', '_')}.png")
    else:
        plt.savefig(f"DynamicDesign/MultiSeg/Obs/{segment.lower().replace(' ', '_')}.png")
    plt.clf()
    plt.close()

def round_to_half(x):
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


def plot_improvement(hour_lst, improvement_pct_lst, improvement_value_lst, goal, segment):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    lns1 = ax.bar(hour_lst, improvement_pct_lst, color="blue", alpha=0.5, label="Pct. Improvement")
    lns2 = ax2.plot(hour_lst, improvement_value_lst, color="red", alpha=0.5, label="Nominal Improvement")
    ax2.scatter(hour_lst, improvement_value_lst, color="red", alpha=0.5)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    if goal in ["Max Revenue", "Min Utility Cost"]:
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    else:
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f} mins"))

    plt.xlabel("Hour")
    plt.tight_layout()

    lns = [lns1] + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="upper left")

    plt.savefig(
        f"DynamicDesign/MultiSeg/Improvements/"
        f"{segment.lower().replace(' ', '_')}_{goal.lower().replace(' ', '_')}.png"
    )
    plt.clf()
    plt.close()


# =========================================================
# Main computation with one common current row per hour
# =========================================================

value_dct = {
    "congestion_current": np.zeros(N_HOURS),
    "congestion_best": np.zeros(N_HOURS),
    "emission_current": np.zeros(N_HOURS),
    "emission_best": np.zeros(N_HOURS),
    "revenue_current": np.zeros(N_HOURS),
    "revenue_best": np.zeros(N_HOURS),
    "utility_cost_current": np.zeros(N_HOURS),
    "utility_cost_best": np.zeros(N_HOURS),
}

# These are the same across segments now, since the baseline is common by hour
common_hour_lst = []
common_toll_avg_by_segment = {seg: [] for seg in SEGMENT_LST}

congestion_improvement_pct_lst = []
congestion_improvement_value_lst = []

emission_improvement_pct_lst = []
emission_improvement_value_lst = []

revenue_improvement_pct_lst = []
revenue_improvement_value_lst = []

utility_cost_improvement_pct_lst = []
utility_cost_improvement_value_lst = []

# Optional: store the optimal toll for each segment under each objective
min_congestion_toll_by_segment = {seg: [] for seg in SEGMENT_LST}
min_emission_toll_by_segment = {seg: [] for seg in SEGMENT_LST}
max_revenue_toll_by_segment = {seg: [] for seg in SEGMENT_LST}
min_utility_cost_toll_by_segment = {seg: [] for seg in SEGMENT_LST}

for hour_idx in range(N_HOURS):
    hour = 7 + hour_idx
    common_hour_lst.append(hour)

    df_design_curr = df_design[df_design["Hour"] == hour].copy()

    # Common current toll vector and current row for this hour
    curr_toll_vec = compute_hourly_actual_toll_vector(df_toll, hour)
    curr_row = get_current_row_from_vector(df_design_curr, curr_toll_vec)

    # Store actual tolls by segment
    for seg_idx, seg in enumerate(SEGMENT_LST):
        common_toll_avg_by_segment[seg].append(curr_toll_vec[seg_idx])

    # Objective values at current row
    curr_travel_time = curr_row["Total Travel Time"]
    curr_emission = curr_row["Total Emission"]
    curr_revenue = curr_row["Total Revenue"]
    curr_utility_cost = curr_row["Total Utility Cost"]

    # Global best values for this hour
    min_travel_time = df_design_curr["Total Travel Time"].min()
    min_emission = df_design_curr["Total Emission"].min()
    max_revenue = df_design_curr["Total Revenue"].max()
    min_utility_cost = df_design_curr["Total Utility Cost"].min()

    value_dct["congestion_current"][hour_idx] = curr_travel_time
    value_dct["congestion_best"][hour_idx] = min_travel_time

    value_dct["emission_current"][hour_idx] = curr_emission
    value_dct["emission_best"][hour_idx] = min_emission

    value_dct["revenue_current"][hour_idx] = curr_revenue
    value_dct["revenue_best"][hour_idx] = max_revenue

    value_dct["utility_cost_current"][hour_idx] = curr_utility_cost
    value_dct["utility_cost_best"][hour_idx] = min_utility_cost

    # Improvements
    congestion_improvement_pct_lst.append(
        (curr_travel_time - min_travel_time) / curr_travel_time * 100 if curr_travel_time != 0 else np.nan
    )
    congestion_improvement_value_lst.append(curr_travel_time - min_travel_time)

    emission_improvement_pct_lst.append(
        (curr_emission - min_emission) / curr_emission * 100 if curr_emission != 0 else np.nan
    )
    emission_improvement_value_lst.append(curr_emission - min_emission)

    revenue_improvement_pct_lst.append(
        (max_revenue - curr_revenue) / curr_revenue * 100 if curr_revenue != 0 else np.nan
    )
    revenue_improvement_value_lst.append(max_revenue - curr_revenue)

    utility_cost_improvement_pct_lst.append(
        (curr_utility_cost - min_utility_cost) / curr_utility_cost * 100 if curr_utility_cost != 0 else np.nan
    )
    utility_cost_improvement_value_lst.append(curr_utility_cost - min_utility_cost)

    # Store segment-wise optimal tolls for price plots
    row_min_congestion = df_design_curr.loc[df_design_curr["Total Travel Time"].idxmin()]
    row_min_emission = df_design_curr.loc[df_design_curr["Total Emission"].idxmin()]
    row_max_revenue = df_design_curr.loc[df_design_curr["Total Revenue"].idxmax()]
    row_min_utility = df_design_curr.loc[df_design_curr["Total Utility Cost"].idxmin()]

    for seg_idx, seg in enumerate(SEGMENT_LST):
        min_congestion_toll_by_segment[seg].append(row_min_congestion[f"Toll {seg_idx}"])
        min_emission_toll_by_segment[seg].append(row_min_emission[f"Toll {seg_idx}"])
        max_revenue_toll_by_segment[seg].append(row_max_revenue[f"Toll {seg_idx}"])
        min_utility_cost_toll_by_segment[seg].append(row_min_utility[f"Toll {seg_idx}"])


# =========================================================
# Segment-specific price plots
# =========================================================
# Assumes plot_hourly_price(...) already exists in your codebase.

for segment_idx, segment in enumerate(SEGMENT_LST):
    segment_short = segment.split("-")[1].split("/")[0].strip()

    toll_avg_lst = common_toll_avg_by_segment[segment]

    # If you still want uncertainty bands from raw toll data, compute them here
    toll_upper_lst = []
    toll_lower_lst = []
    for hour in common_hour_lst:
        df_toll_curr = df_toll[(df_toll["Hour"] == hour) & (df_toll["Segment"] == segment)]
        toll_upper_lst.append(df_toll_curr["Avg_total_toll"].quantile(0.975))
        toll_lower_lst.append(df_toll_curr["Avg_total_toll"].quantile(0.025))

    plot_hourly_price(
        common_hour_lst,
        min_congestion_toll_by_segment[segment],
        toll_avg_lst,
        toll_upper_lst,
        toll_lower_lst,
        "Min Congestion",
        segment_short,
    )
    plot_hourly_price(
        common_hour_lst,
        min_emission_toll_by_segment[segment],
        toll_avg_lst,
        toll_upper_lst,
        toll_lower_lst,
        "Min Emission",
        segment_short,
    )
    plot_hourly_price(
        common_hour_lst,
        max_revenue_toll_by_segment[segment],
        toll_avg_lst,
        toll_upper_lst,
        toll_lower_lst,
        "Max Revenue",
        segment_short,
    )
    plot_hourly_price(
        common_hour_lst,
        min_utility_cost_toll_by_segment[segment],
        toll_avg_lst,
        toll_upper_lst,
        toll_lower_lst,
        "Min Utility Cost",
        segment_short,
    )
    plot_hourly_price(
        common_hour_lst,
        None,
        toll_avg_lst,
        toll_upper_lst,
        toll_lower_lst,
        None,
        segment_short,
    )


# =========================================================
# Improvement plots: these are now common, so plot once
# =========================================================
if PLOT_IMPROVEMENT:

    plot_improvement(
        common_hour_lst,
        congestion_improvement_pct_lst,
        congestion_improvement_value_lst,
        "Min Congestion",
        "total",
    )
    plot_improvement(
        common_hour_lst,
        emission_improvement_pct_lst,
        emission_improvement_value_lst,
        "Min Emission",
        "total",
    )
    plot_improvement(
        common_hour_lst,
        revenue_improvement_pct_lst,
        revenue_improvement_value_lst,
        "Max Revenue",
        "total",
    )
    plot_improvement(
        common_hour_lst,
        utility_cost_improvement_pct_lst,
        utility_cost_improvement_value_lst,
        "Min Utility Cost",
        "total",
    )


    # =========================================================
    # Total improvements
    # =========================================================

    total_congestion_improvement_value_lst = value_dct["congestion_current"] - value_dct["congestion_best"]
    total_congestion_improvement_pct_lst = (
        (value_dct["congestion_current"] - value_dct["congestion_best"]) / value_dct["congestion_current"] * 100
    )

    total_emission_improvement_value_lst = value_dct["emission_current"] - value_dct["emission_best"]
    total_emission_improvement_pct_lst = (
        (value_dct["emission_current"] - value_dct["emission_best"]) / value_dct["emission_current"] * 100
    )

    total_revenue_improvement_value_lst = value_dct["revenue_best"] - value_dct["revenue_current"]
    total_revenue_improvement_pct_lst = (
        (value_dct["revenue_best"] - value_dct["revenue_current"]) / value_dct["revenue_current"] * 100
    )

    total_utility_cost_improvement_value_lst = value_dct["utility_cost_current"] - value_dct["utility_cost_best"]
    total_utility_cost_improvement_pct_lst = (
        (value_dct["utility_cost_current"] - value_dct["utility_cost_best"]) / value_dct["utility_cost_current"] * 100
    )

    # These should now match the earlier improvement lists
    plot_improvement(
        common_hour_lst,
        total_congestion_improvement_pct_lst,
        total_congestion_improvement_value_lst,
        "Min Congestion",
        "total_check",
    )
    plot_improvement(
        common_hour_lst,
        total_emission_improvement_pct_lst,
        total_emission_improvement_value_lst,
        "Min Emission",
        "total_check",
    )
    plot_improvement(
        common_hour_lst,
        total_revenue_improvement_pct_lst,
        total_revenue_improvement_value_lst,
        "Max Revenue",
        "total_check",
    )
    plot_improvement(
        common_hour_lst,
        total_utility_cost_improvement_pct_lst,
        total_utility_cost_improvement_value_lst,
        "Min Utility Cost",
        "total_check",
    )
