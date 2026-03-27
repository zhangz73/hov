import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

TEST_BEGIN_DATE = "2021-05-07"
#SCHEME = "gamma3=2"

## Strategy profile
#df_strategy = pd.read_csv(f"CalibrationResults/Flow/tmp_ratio_{SCHEME}.csv")
df_strategy = pd.read_csv(f"tmp_ratio.csv")
df_strategy["Total"] = df_strategy["Equi 0"] + df_strategy["Equi 1"] + df_strategy["Equi 2"]
for col in ["Equi 0", "Equi 1", "Equi 2", "Target 0", "Target 1", "Target 2"]:
    df_strategy[col] /= df_strategy["Total"]
df_strategy["Date"] = pd.to_datetime(df_strategy["Date"], format = "%Y-%m-%d")
fig, ax = plt.subplots(figsize = (8, 4))
plt.plot(df_strategy["Date"], df_strategy["Equi 0"] * 100, label = "# Single Occup - Equi", color = "blue", alpha = 0.5)
plt.plot(df_strategy["Date"], df_strategy["Target 0"] * 100, linestyle = "dashed", label = "# Single Occup - Obs", color = "blue")
plt.plot(df_strategy["Date"], df_strategy["Equi 1"] * 100, label = "# Carpool 2 - Equi", color = "red", alpha = 0.5)
plt.plot(df_strategy["Date"], df_strategy["Target 1"] * 100, linestyle = "dashed", label = "# Carpool 2 - Obs", color = "red")
plt.plot(df_strategy["Date"], df_strategy["Equi 2"] * 100, label = "# Carpool 3 - Equi", color = "green", alpha = 0.5)
plt.plot(df_strategy["Date"], df_strategy["Target 2"] * 100, linestyle = "dashed", label = "# Carpool 3 - Obs", color = "green")
plt.axvline(x = pd.to_datetime(TEST_BEGIN_DATE, format = "%Y-%m-%d"), color = "red")
plt.xlabel("Date")
plt.ylabel("Fraction of vehicles")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
#plt.savefig(f"CalibrationResults/Plots/daily_ratio_{SCHEME}.png")
plt.savefig("DataVerification/daily_ratio.png")
plt.clf()
plt.close()

## Calibrated flows v.s. actual flows
df_flow = pd.read_csv(f"tmp.csv")
df_flow = df_flow.groupby(["Lane Type", "Segment", "Date", "Hour"]).sum().reset_index()
df_flow = df_flow[df_flow["Flow Target"] > 0]
df_flow["pct_diff"] = (df_flow["Flow Equi"] - df_flow["Flow Target"]) / df_flow["Flow Target"] * 100
df_flow["abs_diff"] = (df_flow["Flow Equi"] - df_flow["Flow Target"])
plt.hist(df_flow["pct_diff"], bins = 100)
plt.axvline(x = 0, color = "red")
plt.xlim(-100, 100)
plt.xlabel("% Difference in equilibrium flow v.s. actual flows")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(f"DataVerification/flow.png")
#plt.savefig(f"DataVerification/flow.png")
plt.clf()
plt.close()
#for hour in df_flow["Hour"].unique():
#    for segment in df_flow["Segment"].unique():
#        for lane in df_flow["Lane Type"].unique():
#            df_sub = df_flow[(df_flow["Hour"] == hour) & (df_flow["Segment"] == segment) & (df_flow["Lane Type"] == lane)]
#            segment_fname = segment.replace("/", "").replace(" ", "")
#            plt.hist(df_sub["pct_diff"], bins = 100)
#            plt.axvline(x = 0, color = "red")
#            plt.xlabel("% Difference in equilibrium flow v.s. actual flows")
#            plt.title(f"Hour: {hour}, Segment: {segment}, {lane}")
#            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
#            plt.tight_layout()
#            plt.savefig(f"DataVerification/DetailedFlow/{hour}_{segment_fname}_{lane}_flow.png")
#            plt.clf()
#            plt.close()

## Calibrated flows v.s. actual flows
df_flow = pd.read_csv("tmp.csv")
df_flow = df_flow.groupby(["Lane Type", "Segment", "Date", "Hour"]).sum().reset_index()
df_flow.loc[df_flow["Lane Type"] == "Ordinary Lane", "Flow Target"] /= 3
df_flow.loc[df_flow["Lane Type"] == "Ordinary Lane", "Flow Equi"] /= 3
df_flow["Data Type"] = df_flow["Date"].apply(lambda x: "Training" if x < TEST_BEGIN_DATE else "Test")

plt.scatter(df_flow[(df_flow["Lane Type"] == "Ordinary Lane") & (df_flow["Data Type"] == "Training")]["Flow Target"], df_flow[(df_flow["Lane Type"] == "Ordinary Lane") & (df_flow["Data Type"] == "Training")]["Flow Equi"], label = "Ordinary Lane (Training)", alpha = 0.2)
plt.scatter(df_flow[(df_flow["Lane Type"] == "HOT Lane") & (df_flow["Data Type"] == "Training")]["Flow Target"], df_flow[(df_flow["Lane Type"] == "HOT Lane") & (df_flow["Data Type"] == "Training")]["Flow Equi"], label = "HOT Lane (Training)", alpha = 0.2)
plt.scatter(df_flow[(df_flow["Lane Type"] == "Ordinary Lane") & (df_flow["Data Type"] == "Test")]["Flow Target"], df_flow[(df_flow["Lane Type"] == "Ordinary Lane") & (df_flow["Data Type"] == "Test")]["Flow Equi"], label = "Ordinary Lane (Test)", alpha = 0.2)
plt.scatter(df_flow[(df_flow["Lane Type"] == "HOT Lane") & (df_flow["Data Type"] == "Test")]["Flow Target"], df_flow[(df_flow["Lane Type"] == "HOT Lane") & (df_flow["Data Type"] == "Test")]["Flow Equi"], label = "HOT Lane (Test)", alpha = 0.2)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel("Observed Flow")
plt.ylabel("Equilibrium Flow")
plt.legend()
plt.tight_layout()
plt.savefig(f"DataVerification/flow_comp.png")
plt.clf()
plt.close()
