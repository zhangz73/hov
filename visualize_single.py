import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

for rho in [0.25, 0.5, 0.75]:
    df_vis = df[df["rho"] == rho]
    plt.plot(df_vis["tau"], df_vis["congestion"], label = f"rho = {rho}")
plt.xlabel("Toll Price")
plt.ylabel("Travel Time")
plt.legend()
plt.savefig("vis.png")
plt.show()

#rho = 0.25
#df_vis = df[df["rho"] == rho]
#for regime in ["A1", "A2", "B"]:
#    df_curr = df_vis[df_vis["regime"] == regime]
#    if df_curr.shape[0] > 0:
#        plt.plot(df_curr["tau"], df_curr["congestion"], label = f"Regime {regime}")
#plt.xlabel("Toll Price")
#plt.ylabel("Travel Time")
#plt.title(f"Rho = {rho}")
#plt.legend()
#plt.show()
