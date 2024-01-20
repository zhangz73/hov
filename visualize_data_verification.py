import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

## Strategy profile
df_strategy = pd.read_csv("tmp_ratio.csv")
df_strategy["Date"] = pd.to_datetime(df_strategy["Date"], format = "%Y-%m-%d")
plt.plot(df_strategy["Date"], df_strategy["Equi 0"], label = "# Single Occup - Equilibrium")
plt.plot(df_strategy["Date"], df_strategy["Target 0"], label = "# Single Occup - Actual")
plt.plot(df_strategy["Date"], df_strategy["Equi 1"], label = "# Carpool 2 - Equilibrium")
plt.plot(df_strategy["Date"], df_strategy["Target 1"], label = "# Carpool 2 - Actual")
plt.plot(df_strategy["Date"], df_strategy["Equi 2"], label = "# Carpool 3 - Equilibrium")
plt.plot(df_strategy["Date"], df_strategy["Target 2"], label = "# Carpool 3 - Actual")
plt.xlabel("Date")
plt.ylabel("Number of vehicles")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.tight_layout()
plt.savefig("DataVerification/daily_ratio.png")
plt.clf()
plt.close()

## Calibrated flows v.s. actual flows
df_flow = pd.read_csv("tmp.csv")
df_flow["pct_diff"] = (df_flow["Flow Equi"] - df_flow["Flow Target"]) / df_flow["Flow Target"] * 100
plt.hist(df_flow["pct_diff"], bins = 100)
plt.xlabel("% Difference in equilibrium flow v.s. actual flows")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(f"DataVerification/flow.png")
plt.clf()
plt.close()
