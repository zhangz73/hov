import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## Strategy profile
WINDOW_SIZE = 5
df_strategy = pd.read_csv("data/df_date_profile.csv")
df_strategy = df_strategy[~df_strategy["Date"].isin(["2021-03-31", "2021-06-30"])]
df_strategy["Date"] = pd.to_datetime(df_strategy["Date"], format = "%Y-%m-%d")
df_strategy["sigma_2ratio_equi"] = df_strategy["sigma_2ratio_equi"].rolling(WINDOW_SIZE).mean()
df_strategy["sigma_3ratio_equi"] = df_strategy["sigma_3ratio_equi"].rolling(WINDOW_SIZE).mean()
plt.plot(df_strategy["Date"], df_strategy["sigma_2ratio_equi"], label = "% Carpool 2 - Equilibrium")
plt.plot(df_strategy["Date"], df_strategy["Sigma_2ratio"], label = "% Carpool 2 - Actual")
plt.plot(df_strategy["Date"], df_strategy["sigma_3ratio_equi"], label = "% Carpool 3 - Equilibrium")
plt.plot(df_strategy["Date"], df_strategy["Sigma_3ratio"], label = "% Carpool 3 - Actual")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.tight_layout()
plt.savefig("DataVerification/strategy.png")
plt.clf()
plt.close()

## Demand change over time
def q1(x):
    return x.quantile(0.025)

def q3(x):
    return x.quantile(0.975)
    
df_demand = pd.read_csv("data/df_meta_w_demand.csv")
df_demand = df_demand[["Hour", "Demand"]].groupby("Hour").agg({"Demand": ["mean", q1, q3]}).reset_index().sort_values("Hour")
df_demand.columns = ["Hour", "Demand", "Demand Lower", "Demand Upper"]
plt.scatter(df_demand["Hour"], df_demand["Demand"], color = "blue")
plt.plot(df_demand["Hour"], df_demand["Demand"], color = "blue")
plt.fill_between(df_demand["Hour"], df_demand["Demand Lower"], df_demand["Demand Upper"], color = "blue", alpha = 0.2)
plt.xlabel("Time of Day")
plt.ylabel("Demand")
plt.savefig("DataVerification/demand.png")
plt.clf()
plt.close()

## Calibrated flows v.s. actual flows
df_demand = pd.read_csv("data/df_meta_w_demand.csv")
hour = 15
df_demand = df_demand[df_demand["Hour"] == hour]
df_demand = df_demand.sort_values(["Date", "Hour"], ascending = False)
df_demand["HOV Flow Equi"] = df_demand["Demand"] * (df_demand["Sigma_toll"] + 1/2 * df_demand["Sigma_pool2"] + 1/3 * df_demand["Sigma_pool3"])
df_demand["Ordinary Flow Equi"] = df_demand["Demand"] * df_demand["Sigma_o"]
df_demand["Time"] = pd.to_datetime(df_demand["Date"] + " " + df_demand["Hour"].astype(str) + ":00", format = "%Y-%m-%d %H")
plt.plot(df_demand["Time"], df_demand["HOV Flow Equi"], label = "HOV Flow - Equilibrium")
plt.plot(df_demand["Time"], df_demand["HOV Flow"], label = "HOV Flow - Actual")
plt.plot(df_demand["Time"], df_demand["Ordinary Flow Equi"], label = "Ordinary Flow - Equilibrium")
plt.plot(df_demand["Time"], df_demand["Ordinary Flow"], label = "Ordinary Flow - Actual")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.tight_layout()
plt.savefig(f"DataVerification/flow_{hour}.png")
plt.clf()
plt.close()
