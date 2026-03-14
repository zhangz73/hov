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

## Plot ratio of people going to ordinary lane as tau increases
#for rho in [0.25, 0.5, 0.75]:
#    df_vis = df[df["rho"] == rho]
#    df_vis = df_vis[df_vis["regime"] == "B"]
#    x3 = np.array(df_vis["x3"])
#    x1 = np.array(df_vis["x1"])
#    gamma = -(x3[1:] - x3[:-1]) / (x1[1:] - x1[:-1])
#    plt.plot(df_vis.iloc[:-1]["tau"], gamma, label = f"rho = {rho}")
#plt.xlabel("Toll Price")
#plt.ylabel("Fraction of People Moving to Ordinary Lane")
#plt.legend()
#plt.savefig("vis.png")
#plt.show()

#for rho in [0.25, 0.5, 0.75]:
#    df_vis = df[df["rho"] == rho]
#    plt.plot(df_vis["x2"], df_vis["x1"], label = f"rho = {rho}")
#    plt.axline((0, rho), slope = -1.25, color = "red", alpha = 0.3)
#plt.xlabel("% People Carpool")
#plt.ylabel("% People Paying Tolls")
#plt.legend()
##plt.savefig("vis_o.png")
#plt.show()

#for rho in [0.25, 0.5, 0.75]:
#    df_vis = df[df["rho"] == rho]
#    plt.plot(df_vis["tau"], df_vis["x3"], label = f"rho = {rho}")
#plt.xlabel("Toll Price")
#plt.ylabel("% People Using Ordinary Lane")
#plt.legend()
##plt.savefig("vis_o.png")
#plt.show()

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

#rho = 0.25
#A = 2.5
#df_vis = df[df["rho"] == rho]
#lb = (1 - rho) / (1 - rho + A * rho)
#ub = (1 - rho + A * rho * (1 - rho) ** 0.5) / (1 - rho + A * rho)
#x = df_vis["x3"]
#y = df_vis["x2"]
#x_data = np.linspace(lb, ub, 100)
#lo = (x_data - (1 - rho)) ** 2 / ((1 - rho) * (1 - 1/A) * (1 - x_data))
#hi = 1 - x_data
#x_pos = np.argmax(hi <= lo)
#plt.vlines(x = lb, ymin = lo[0], ymax = hi[0], color = "blue")
##plt.axvline(x = ub, color = "blue")
#plt.plot(x_data[:x_pos], lo[:x_pos], color = "blue")
#plt.plot(x_data[:x_pos], hi[:x_pos], color = "blue")
#plt.fill_between(x_data[:x_pos], lo[:x_pos], hi[:x_pos], color = "blue", alpha = 0.3)
#plt.scatter(x, y, color = "red")
#plt.xlabel("Fraction of People Ordinary Lane")
#plt.ylabel("Fraction of People Carpool")
#plt.title(f"Rho = {rho}")
#plt.savefig("region.png")
#plt.show()
