import numpy as np
import pandas as pd

df = pd.read_csv("toll_design_multiseg_hour=17_multi-rho.csv")
df_toll = pd.read_csv("data/df_toll.csv")
df_toll = df_toll[df_toll["Hour"] == 17][["Segment", "Avg_total_toll"]]
df_toll_avg = df_toll.groupby("Segment").mean()
N_POP = 25837
df["Total Travel Time"] /= N_POP
df["Total Emission"] /= N_POP
df["Total Utility Cost"] /= N_POP

with open("toll.txt", "w") as f:
    for rho in [0.25, 0.5, 0.75]:
        df_curr = df[df["Rho"] == rho]
        for feat in ["Total Travel Time", "Total Emission", "Total Revenue", "Total Utility Cost"]:
            if feat != "Total Revenue":
                row = df_curr[df_curr[feat] == df_curr[feat].min()].iloc[0]
            else:
                row = df_curr[df_curr[feat] == df_curr[feat].max()].iloc[0]
            msg = ""
            for i in range(5):
                msg += str(row[f"Toll {i}"]) + " "
            msg += str(row[feat])
            f.write(f"rho = {rho}, {feat}: {msg}\n")

    f.write("\n")
    f.write(df_toll_avg.to_string())
