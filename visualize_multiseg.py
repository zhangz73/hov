import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SEGMENT_LST = ['3420 - Auto Mall NB', '3430 - Mowry NB', '3440 - Decoto/84 NB', '3450 - Whipple NB', '3460 - Hesperian/238 NB']
df_design = pd.read_csv("toll_design_multiseg.csv")
df_design = df_design[df_design["Rho"] == 0.25]
## Date, Hour, Segment, Avg_total_toll
df_toll = pd.read_csv("data/df_toll.csv")

def plot_hourly_price(hour_lst, toll_design_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, goal, segment):
    plt.plot(hour_lst, toll_design_lst, color = "red", label = "Optimal Toll Price")
    plt.scatter(hour_lst, toll_avg_lst, color = "blue", label = "Current Toll Price")
    plt.fill_between(hour_lst, toll_lower_lst, toll_upper_lst, color = "blue", alpha = 0.2)
#    plt.gcf().axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
#    plt.gcf().autofmt_xdate()
    plt.xlabel("Time of Day")
    plt.ylabel(f"{segment} - {goal}")
    plt.legend(loc = "upper left")
    plt.savefig(f"DynamicDesign/MultiSeg/{segment.lower().replace(' ', '_')}_{goal.lower().replace(' ', '_')}.png")
    plt.clf()
    plt.close()

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
    for hour_idx in range(15):
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
    segment_short = segment.split("-")[1].split("/")[0].strip()
    plot_hourly_price(hour_lst, min_congestion_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Min Congestion", segment_short)
    plot_hourly_price(hour_lst, min_emission_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Min Emission", segment_short)
    plot_hourly_price(hour_lst, max_revenue_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Max Revenue", segment_short)
    plot_hourly_price(hour_lst, min_utility_cost_toll_lst, toll_avg_lst, toll_upper_lst, toll_lower_lst, "Min Utility Cost", segment_short)
