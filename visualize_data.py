import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv("data/df_hourly_avg.csv")
#plt.plot(df["Hour"], df["HOV Flow"], label = "HOV Flow")
#plt.plot(df["Hour"], df["Ordinary Flow"] / 3, label = "Ordinary Flow")
#plt.xlabel("Hour")
#plt.ylabel("Average Flow")
#plt.legend()
#plt.savefig("flow.png")
#plt.clf()
#plt.close()
#
#plt.plot(df["Hour"], df["HOV Travel Time"], label = "HOV Travel Time")
#plt.plot(df["Hour"], df["Ordinary Travel Time"], label = "Ordinary Travel Time")
#plt.xlabel("Hour")
#plt.ylabel("Average Travel Time")
#plt.legend()
#plt.savefig("travel_time.png")
#plt.clf()
#plt.close()

def q1(x):
    return x.quantile(0.025)

def q3(x):
    return x.quantile(0.975)

df_demand = pd.read_csv("data/df_meta_w_demand.csv")
df_hov = df_demand[["Hour", "HOV Travel Time"]].groupby("Hour").agg({"HOV Travel Time": ["mean", q1, q3]}).reset_index().sort_values("Hour")
df_hov.columns = ["Hour", "HOV Travel Time", "HOV Travel Time Lower", "HOV Travel Time Upper"]
df_o = df_demand[["Hour", "Ordinary Travel Time"]].groupby("Hour").agg({"Ordinary Travel Time": ["mean", q1, q3]}).reset_index().sort_values("Hour")
df_o.columns = ["Hour", "Ordinary Travel Time", "Ordinary Travel Time Lower", "Ordinary Travel Time Upper"]
plt.scatter(df_hov["Hour"], df_hov["HOV Travel Time"], color = "blue")
plt.plot(df_hov["Hour"], df_hov["HOV Travel Time"], color = "blue", label = "HOT Travel Time")
plt.fill_between(df_hov["Hour"], df_hov["HOV Travel Time Lower"], df_hov["HOV Travel Time Upper"], color = "blue", alpha = 0.2)
plt.scatter(df_o["Hour"], df_o["Ordinary Travel Time"], color = "orange")
plt.plot(df_o["Hour"], df_o["Ordinary Travel Time"], color = "orange", label = "Ordinary Travel Time")
plt.fill_between(df_o["Hour"], df_o["Ordinary Travel Time Lower"], df_o["Ordinary Travel Time Upper"], color = "orange", alpha = 0.2)
plt.xlabel("Hour")
plt.ylabel("Travel Time")
plt.legend(loc = "upper left")
plt.savefig("travel_time.png")
plt.clf()
plt.close()

df_demand = pd.read_csv("data/df_meta_w_demand.csv")
df_hov = df_demand[["Hour", "HOV Flow"]].groupby("Hour").agg({"HOV Flow": ["mean", q1, q3]}).reset_index().sort_values("Hour")
df_hov.columns = ["Hour", "HOV Flow", "HOV Flow Lower", "HOV Flow Upper"]
df_o = df_demand[["Hour", "Ordinary Flow"]].groupby("Hour").agg({"Ordinary Flow": ["mean", q1, q3]}).reset_index().sort_values("Hour")
df_o.columns = ["Hour", "Ordinary Flow", "Ordinary Flow Lower", "Ordinary Flow Upper"]
plt.scatter(df_hov["Hour"], df_hov["HOV Flow"], color = "blue")
plt.plot(df_hov["Hour"], df_hov["HOV Flow"], color = "blue", label = "HOT Flow")
plt.fill_between(df_hov["Hour"], df_hov["HOV Flow Lower"], df_hov["HOV Flow Upper"], color = "blue", alpha = 0.2)
plt.scatter(df_o["Hour"], df_o["Ordinary Flow"], color = "orange")
plt.plot(df_o["Hour"], df_o["Ordinary Flow"], color = "orange", label = "Ordinary Flow")
plt.fill_between(df_o["Hour"], df_o["Ordinary Flow Lower"], df_o["Ordinary Flow Upper"], color = "orange", alpha = 0.2)
plt.xlabel("Hour")
plt.ylabel("Flow")
plt.legend(loc = "upper left")
plt.savefig("flow.png")
plt.clf()
plt.close()
