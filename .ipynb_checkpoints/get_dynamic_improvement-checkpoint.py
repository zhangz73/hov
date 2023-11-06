import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df = pd.read_csv("time_dynamic_design_all.csv")
hour_lst = []
for t in range(5, 20):
    hour_lst += [t] * 150
df["Hour"] = hour_lst
df_data = pd.read_csv("data/df_hourly_avg.csv")

rho = 0.25
granularity = 0.1
travel_time_lst = []
emission_lst = []
revenue_lst = []
utility_cost_lst = []
hour_lst = np.array(df_data["Hour"])
for t in hour_lst:
    df_tmp = df[(df["Hour"] == t)]
    toll = df_data[df_data["Hour"] == t].iloc[0]["Toll"]
    nearest_row = df_tmp.iloc[(df_tmp['Toll Price']-toll).abs().argsort().iloc[0]]
    curr_travel_time = nearest_row["Total Travel Time"]
    curr_emission = nearest_row["Total Emission"]
    curr_revenue = nearest_row["Total Revenue"]
    curr_utility_cost = nearest_row["Total Utility Cost"]
    min_travel_time = df_tmp["Total Travel Time"].min()
    min_emission = df_tmp["Total Emission"].min()
    max_revenue = df_tmp["Total Revenue"].max()
    min_utility_cost = df_tmp["Total Utility Cost"].min()
    travel_time_lst.append((curr_travel_time - min_travel_time) / curr_travel_time * 100)
    emission_lst.append((curr_emission - min_emission) / curr_emission * 100)
    revenue_lst.append((max_revenue - curr_revenue) / curr_revenue * 100)
    utility_cost_lst.append((curr_utility_cost - min_utility_cost) / curr_utility_cost * 100)

plt.bar(hour_lst, travel_time_lst)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel("Hour")
plt.title("Congestion Improvement")
plt.savefig("DynamicDesign/dynamic_congestion_improvement.png")
plt.clf()
plt.close()

plt.bar(hour_lst, emission_lst)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel("Hour")
plt.title("Emission Improvement")
plt.savefig("DynamicDesign/dynamic_emission_improvement.png")
plt.clf()
plt.close()

plt.bar(hour_lst, revenue_lst)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel("Hour")
plt.title("Revenue Improvement")
plt.savefig("DynamicDesign/dynamic_revenue_improvement.png")
plt.clf()
plt.close()

plt.bar(hour_lst, utility_cost_lst)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel("Hour")
plt.title("Utility Cost Improvement")
plt.savefig("DynamicDesign/dynamic_utility_cost_improvement.png")
plt.clf()
plt.close()