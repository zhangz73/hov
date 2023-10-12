import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def compute_pareto_front(df, colname, xlabel, fname):
    df = df.sort_values("Total Revenue", ascending = True).copy()
    pareto = []
    pareto_sep = []
    for i in range(df.shape[0]):
        is_pareto = 1
        is_pareto_sep = 1
        feat_i = df.iloc[i][colname]
        revenue_i = df.iloc[i]["Total Revenue"]
        rho_i = df.iloc[i]["HOT Capacity"]
        for j in range(i + 1, df.shape[0]):
            feat_j = df.iloc[j][colname]
            revenue_j = df.iloc[j]["Total Revenue"]
            rho_j = df.iloc[j]["HOT Capacity"]
            if feat_j < feat_i and revenue_j > revenue_i:
                is_pareto = 0
                if rho_i == rho_j:
                    is_pareto_sep = 0
        pareto.append(is_pareto)
        pareto_sep.append(is_pareto_sep)
    df["pareto"] = pareto
    df["pareto_sep"] = pareto_sep
    
    df_pareto = df[df["pareto"] == 1]
#    plt.scatter(df_pareto[colname], df_pareto["Total Revenue"])
#    plt.plot(df_pareto[colname], df_pareto["Total Revenue"])
#    plt.xlabel(xlabel)
#    plt.ylabel("Total Revenue Gathered From Tolls ($)")
#    #plt.title("Pareto Front")
#    plt.savefig(f"{fname}.png")
#    plt.clf()
#    plt.close()

    for rho in [0.25, 0.5, 0.75]:
        df_sub = df[(df["HOT Capacity"] == rho) & (df["pareto_sep"] == 1)]
        plt.scatter(df_sub[colname], df_sub["Total Revenue"], label = f"$\\rho = {rho}$")
        plt.plot(df_sub[colname], df_sub["Total Revenue"])
#    plt.plot(df_pareto[colname], df_pareto["Total Revenue"], color = "black", alpha = 0.5, label = "Combined", linestyle = "--")
    plt.xlabel(xlabel)
    plt.ylabel("Total Revenue Gathered From Tolls ($)")
    #plt.title("Pareto Front")
    plt.legend()
    plt.savefig(f"pareto_{fname}_sep.png")
    plt.clf()
    plt.close()

df = pd.read_csv("opt_3d_results.csv")
lst = list(np.arange(0, 15, 0.1)[:])
df = df[df["Toll Price"].isin(lst)]
rho_vals = [0.25, 0.5, 0.75]
## Visualize price curves
for rho in rho_vals:
    plt.plot(df[df["HOT Capacity"] == rho]["Toll Price"], df[df["HOT Capacity"] == rho]["Total Travel Time"], label = f"$\\rho = {rho}$")
    df_tmp = df[df["HOT Capacity"] == rho]
    latency_lst = np.array(df_tmp["Total Travel Time"])
    toll_lst = np.array(df_tmp["Toll Price"])
    print(f"rho = {rho}", "Min Latency At: tau =", toll_lst[np.argmin(latency_lst)], "Min Latency =", np.min(latency_lst))
#plt.plot(tau_lst, total_emission_lst, label = "Total Emission")
plt.xlabel("Toll Price ($)")
plt.ylabel("Average Traffic Time Per Traveler (Minutes)")
plt.legend()
plt.savefig("3d_latency_vs_toll.png")
plt.clf()
plt.close()

for rho in rho_vals:
    plt.plot(df[df["HOT Capacity"] == rho]["Toll Price"], df[df["HOT Capacity"] == rho]["Total Emission"], label = f"$\\rho = {rho}$")
    df_tmp = df[df["HOT Capacity"] == rho]
    emission_lst = np.array(df_tmp["Total Emission"])
    toll_lst = np.array(df_tmp["Toll Price"])
    print(f"rho = {rho}", "Min Emission At: tau =", toll_lst[np.argmin(emission_lst)], "Min Emission =", np.min(emission_lst))
plt.xlabel("Toll Price ($)")
plt.ylabel("Average Emission Per Traveler (Minutes)")
plt.legend()
plt.savefig("3d_emission_vs_toll.png")
plt.clf()
plt.close()

for rho in rho_vals:
    plt.plot(df[df["HOT Capacity"] == rho]["Toll Price"], df[df["HOT Capacity"] == rho]["Total Revenue"], label = f"$\\rho = {rho}$")
    df_tmp = df[df["HOT Capacity"] == rho]
    revenue_lst = np.array(df_tmp["Total Revenue"])
    toll_lst = np.array(df_tmp["Toll Price"])
    print(f"rho = {rho}", "Max Revenue At: tau =", toll_lst[np.argmax(revenue_lst)], "Max Revenue =", np.max(revenue_lst))
plt.xlabel("Toll Price ($)")
plt.ylabel("Total Revenue Gathered From Tolls ($)")
#plt.title(f"Max Revenue Achieved At: tau = {tau_lst[np.argmax(total_revenue_lst)]}")
plt.legend()
plt.savefig("3d_revenue_vs_toll.png")
plt.clf()
plt.close()

## Compute pareto front
compute_pareto_front(df, "Total Travel Time", "Average Traffic Time Per Traveler (Minutes)", "latency")
compute_pareto_front(df, "Total Emission", "Average Emission Per Traveler (Minutes)", "emission")

## Visualize dynamic pricing
df_dynamic = pd.read_csv("time_dynamic_design.csv")
df_data = pd.read_csv("data/all_tolls.csv")
df_data = df_data[["Time", "Toll"]]
df_data["Hour"] = df_data["Time"].apply(lambda x: int(x.split(":")[0]))
df_all = df_dynamic.merge(df_data, on = "Hour")
feat_lst = ["Min Congestion", "Min Emission", "Max Revenue"]
rho = 0.25
for feat in feat_lst:
    time_lst = pd.to_datetime(df_all[df_all["Rho"] == rho]["Time"], format="%H:%M") # np.array(df_all["Hour"]) #
    opt_toll = np.array(df_all[df_all["Rho"] == rho][f"{feat} Toll"]) #np.repeat(np.array(df_dynamic[df_dynamic["Rho"] == rho][f"{feat} Toll"]), 60 // TIME_FREQ)
    curr_toll = df_all[df_all["Rho"] == rho]["Toll"]
    plt.plot(time_lst, opt_toll, label = "Optimal Toll Price", color = "red")
    plt.scatter(time_lst, curr_toll, label = "Current Toll Price", color = "blue")
    plt.gcf().axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time of Day")
    plt.ylabel(f"Optimal Toll Price For {feat}")
    plt.legend()
    plt.savefig(f"DynamicDesign/{feat.lower().replace(' ', '_')}.png")
    plt.clf()
    plt.close()
