import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

SEGMENT_LST = ['3420 - Auto Mall NB', '3430 - Mowry NB', '3440 - Decoto/84 NB', '3450 - Whipple NB', '3460 - Hesperian/238 NB']
df_design = pd.read_csv("../toll_design_multiseg.csv")
df_design = df_design[df_design["Rho"] == 0.25]
## Date, Hour, Segment, Avg_total_toll
df_toll = pd.read_csv("data/df_toll.csv")

INT_GRID = 10
N_POP = 1#INT_GRID ** 3 # 24546
df_design["Total Travel Time"] /= N_POP
df_design["Total Emission"] /= N_POP
df_design["Total Utility Cost"] /= N_POP

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

def plot_improvement(hour_lst, improvement_pct_lst, improvement_value_lst, goal, segment):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    lns1 = ax.bar(hour_lst, improvement_pct_lst, color = "blue", alpha = 0.5, label = "Pct. Improvement")
    lns2 = ax2.plot(hour_lst, improvement_value_lst, color = "red", alpha = 0.5, label = "Nominal Improvement")
    ax2.scatter(hour_lst, improvement_value_lst, color = "red", alpha = 0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    if goal in ["Max Revenue", "Min Utility Cost"]:
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    else:
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f} mins"))
    plt.xlabel("Hour")
    plt.tight_layout()
    lns = [lns1]+lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc = "upper left")
    plt.savefig(f"DynamicDesign/MultiSeg/Improvements/{segment.lower().replace(' ', '_')}_{goal.lower().replace(' ', '_')}.png")
    plt.clf()
    plt.close()

N_HOURS = 15
value_dct = {"congestion_current": np.zeros(N_HOURS), "congestion_best": np.zeros(N_HOURS), "emission_current": np.zeros(N_HOURS), "emission_best": np.zeros(N_HOURS), "revenue_current": np.zeros(N_HOURS), "revenue_best": np.zeros(N_HOURS), "utility_cost_current": np.zeros(N_HOURS), "utility_cost_best": np.zeros(N_HOURS)}
for segment_idx in range(len(SEGMENT_LST)):
    segment = SEGMENT_LST[segment_idx]
    hour_lst = []
    min_congestion_toll_lst = []
    min_emission_toll_lst = []
    max_revenue_toll_lst = []
    min_utility_cost_toll_lst = []
    toll_avg_lst = []
    toll_upper_lst = []
    toll_lower_lst = []
    congestion_improvement_pct_lst = []
    congestion_improvement_value_lst = []
    emission_improvement_pct_lst = []
    emission_improvement_value_lst = []
    revenue_improvement_pct_lst = []
    revenue_improvement_value_lst = []
    utility_cost_improvement_pct_lst = []
    utility_cost_improvement_value_lst = []
    for hour_idx in range(N_HOURS):
        hour = 5 + hour_idx
        hour_lst.append(hour)
        df_design_curr = df_design[df_design["Hour"] == hour]
        min_congestion_toll = df_design_curr[df_design_curr["Total Travel Time"] == df_design_curr["Total Travel Time"].min()].iloc[0][f"Toll {segment_idx}"]
        min_emission_toll = df_design_curr[df_design_curr["Total Emission"] == df_design_curr["Total Emission"].min()].iloc[0][f"Toll {segment_idx}"]
        max_revenue_toll = df_design_curr[df_design_curr["Total Revenue"] == df_design_curr["Total Revenue"].max()].iloc[0][f"Toll {segment_idx}"]
        min_utility_cost_toll = df_design_curr[df_design_curr["Total Utility Cost"] == df_design_curr["Total Utility Cost"].min()].iloc[0][f"Toll {segment_idx}"]
        df_toll_curr = df_toll[(df_toll["Hour"] == hour) & (df_toll["Segment"] == segment)]
        toll_avg = df_toll_curr["Avg_total_toll"].mean()
        toll_upper = df_toll_curr["Avg_total_toll"].quantile(0.025)
        toll_lower = df_toll_curr["Avg_total_toll"].quantile(0.975)
        min_congestion_toll_lst.append(min_congestion_toll)
        min_emission_toll_lst.append(min_congestion_toll)
        max_revenue_toll_lst.append(max_revenue_toll)
        min_utility_cost_toll_lst.append(min_utility_cost_toll)
        toll_avg_lst.append(toll_avg)
        toll_upper_lst.append(toll_upper)
        toll_lower_lst.append(toll_lower)
        nearest_row = df_design_curr.iloc[(df_design_curr[f"Toll {segment_idx}"] - toll_avg).abs().argsort().iloc[0]]
        curr_travel_time = nearest_row["Total Travel Time"]
        curr_emission = nearest_row["Total Emission"]
        curr_revenue = nearest_row["Total Revenue"]
        curr_utility_cost = nearest_row["Total Utility Cost"]
        min_travel_time = df_design_curr["Total Travel Time"].min()
        min_emission = df_design_curr["Total Emission"].min()
        max_revenue = df_design_curr["Total Revenue"].max()
        min_utility_cost = df_design_curr["Total Utility Cost"].min()
        value_dct["congestion_current"][hour_idx] += curr_travel_time
        value_dct["congestion_best"][hour_idx] += min_travel_time
        value_dct["emission_current"][hour_idx] += curr_emission
        value_dct["emission_best"][hour_idx] += min_emission
        value_dct["revenue_current"][hour_idx] += curr_revenue
        value_dct["revenue_best"][hour_idx] += max_revenue
        value_dct["utility_cost_current"][hour_idx] += curr_utility_cost
        value_dct["utility_cost_best"][hour_idx] += min_utility_cost
        congestion_improvement_pct_lst.append((curr_travel_time - min_travel_time) / curr_travel_time * 100)
        emission_improvement_pct_lst.append((curr_emission - min_emission) / curr_emission * 100)
        revenue_improvement_pct_lst.append((max_revenue - curr_revenue) / curr_revenue * 100)
        utility_cost_improvement_pct_lst.append((curr_utility_cost - min_utility_cost) / curr_utility_cost * 100)
        congestion_improvement_value_lst.append((curr_travel_time - min_travel_time))
        emission_improvement_value_lst.append((curr_emission - min_emission))
        revenue_improvement_value_lst.append((max_revenue - curr_revenue))
        utility_cost_improvement_value_lst.append((curr_utility_cost - min_utility_cost))
    segment_short = segment.split("-")[1].split("/")[0].strip()
    plot_hourly_price(hour_lst, min_congestion_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Min Congestion", segment_short)
    plot_hourly_price(hour_lst, min_emission_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Min Emission", segment_short)
    plot_hourly_price(hour_lst, max_revenue_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Max Revenue", segment_short)
    plot_hourly_price(hour_lst, min_utility_cost_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Min Utility Cost", segment_short)
    plot_hourly_price(hour_lst, None, toll_avg_lst, toll_upper_lst, toll_lower_lst, None, segment_short)
    plot_improvement(hour_lst, congestion_improvement_pct_lst, congestion_improvement_value_lst, "Min Congestion", segment_short)
    plot_improvement(hour_lst, emission_improvement_pct_lst, emission_improvement_value_lst, "Min Emission", segment_short)
    plot_improvement(hour_lst, revenue_improvement_pct_lst, revenue_improvement_value_lst, "Max Revenue", segment_short)
    plot_improvement(hour_lst, utility_cost_improvement_pct_lst, utility_cost_improvement_value_lst, "Min Utility Cost", segment_short)

## Compute total improvements
total_congestion_improvement_value_lst = value_dct["congestion_current"] - value_dct["congestion_best"]
total_congestion_improvement_pct_lst = (value_dct["congestion_current"] - value_dct["congestion_best"]) / value_dct["congestion_current"] * 100
total_emission_improvement_value_lst = value_dct["emission_current"] - value_dct["emission_best"]
total_emission_improvement_pct_lst = (value_dct["emission_current"] - value_dct["emission_best"]) / value_dct["emission_current"] * 100
total_revenue_improvement_value_lst = value_dct["revenue_best"] - value_dct["revenue_current"]
total_revenue_improvement_pct_lst = (value_dct["revenue_best"] - value_dct["revenue_current"]) / value_dct["revenue_current"] * 100
total_utility_cost_improvement_value_lst = value_dct["utility_cost_current"] - value_dct["utility_cost_best"]
total_utility_cost_improvement_pct_lst = (value_dct["utility_cost_current"] - value_dct["utility_cost_best"]) / value_dct["utility_cost_current"] * 100
plot_improvement(hour_lst, total_congestion_improvement_pct_lst, total_congestion_improvement_value_lst, "Min Congestion", "total")
plot_improvement(hour_lst, total_emission_improvement_pct_lst, total_emission_improvement_value_lst, "Min Emission", "total")
plot_improvement(hour_lst, total_revenue_improvement_pct_lst, total_revenue_improvement_value_lst, "Max Revenue", "total")
plot_improvement(hour_lst, total_utility_cost_improvement_pct_lst, total_utility_cost_improvement_value_lst, "Min Utility Cost", "total")
