import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/df_wide.csv")
df = df[~df["Date"].isin(["2021-03-31", "2021-04-23", "2021-04-26", "2021-06-30"])]
attr_lst = ["HOV Flow", "Ordinary Flow", "HOV Travel Time", "Ordinary Travel Time", "Avg_total_toll"]
df = df.sort_values(by = ["Date", "Hour"])
hour_lst = df["Hour"].unique()
#for attr in attr_lst:
#    for hour in hour_lst:
#        cols = [x for x in df.columns if x.startswith(attr)]
#        name = attr.lower().replace(" ", "_")
#        for col in cols:
#            plt.plot(df[df["Hour"] == hour][col], label = col)
#        plt.legend()
#        plt.savefig(f"DataTrend/{name}_{hour}.png")
#        plt.clf()
#        plt.close()

df["demand"] = 0
for col in df.columns:
    if "Flow" in col:
        df["demand"] += df[col]
#for hour in hour_lst:
#    plt.plot(df[df["Hour"] == hour]["demand"])
#    plt.savefig(f"DataTrend/demand_{hour}.png")
#    plt.clf()
#    plt.close()

df_demand = df[["Date", "demand"]].groupby(["Date"]).sum().reset_index().sort_values("Date")
slope_intercept = np.polyfit(np.arange(df_demand.shape[0]), df_demand["demand"], 1)
plt.plot(df_demand["demand"])
plt.axline((0, slope_intercept[1]), slope = slope_intercept[0], color = "red")
plt.savefig(f"DataTrend/demand.png")
plt.clf()
plt.close()
