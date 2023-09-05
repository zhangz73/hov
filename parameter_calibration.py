import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch

RELEVANT_TIME = ("16:30", "17:30")
RELEVANT_ZONE = "3460 - Hesperian/238 NB"
RELEVANT_STATIONS = [400488, 401561, 400611, 400928, 400284, 400041, 408133, 408135, 417665, 412637, 417666, 408134, 400685, 401003, 400898, 400275, 400939, 400180, 400529, 400990, 400515, 400252]

## Load toll price data
df = pd.read_csv("NB_042021-062021.csv")
#df = df[pd.to_numeric(df["Segment_Toll"], errors='coerce').notnull()]
df = df[pd.to_numeric(df["Zone_Toll"], errors='coerce').notnull()]
df = df.dropna(subset = ["Zone_Toll"])[["dtMsgStartTime2", "siZoneID", "iPlazaID", "Zone_Toll", "Segment_Toll"]]
df = df.fillna(0)
df = df[df["siZoneID"] == RELEVANT_ZONE]
df["Time"] = pd.to_datetime(df["dtMsgStartTime2"]).dt.strftime("%H:%M")
df = df[(df["Time"] >= RELEVANT_TIME[0]) & (df["Time"] <= RELEVANT_TIME[1])]
df["Toll"] = df["Zone_Toll"].astype(float)
df = df[["Time", "Toll"]].groupby(["Time"]).mean().reset_index().sort_values("Time")


#plt.plot(pd.to_datetime(df["Time"], format="%H:%M"), df["Toll"])
#plt.gcf().autofmt_xdate()
#plt.show()

## Load population data
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
sigma_2over1 = df_pop["TwoPeople"].sum() / df_pop["Single"].sum()
sigma_3over1 = df_pop["ThreePlus"].sum() / df_pop["Single"].sum()

## Load station data
#df_station = pd.read_csv("d04_text_meta_2021_03_19.txt", sep = "\t")
#df_station = df_station[(df_station["Fwy"] == 880) & (df_station["Dir"] == "N") & (df_station["Type"] == "ML")]
#df_station.to_csv("station_meta.csv", index = False)
#print(df_station)
df_station = pd.read_csv("station_meta.csv")

## Load flow data
N = 8
num_lanes = 4
lane_names = list(itertools.chain(*[[f"Lane {i} Samples", f"Lane {i} Flow", f"Lane {i} Avg Occ", f"Lane {i} Avg Speed", f"Lane {i} Observed"] for i in range(N)]))
names = ["Timestamp", "Station", "District", "Freeway", "Direction", "LaneType", "StationLength", "Samples", "% Observed", "Total Flow", "Avg Occupancy", "Avg Speed"] + lane_names
df_flow = pd.read_csv("d04_text_station_5min_2021_04_01.txt", header = None, names = names)
df_flow = df_flow[(df_flow["Direction"] == "N") & (df_flow["Freeway"] == 880) & (df_flow["LaneType"] == "ML")]
df_flow = df_flow[df_flow["Station"].isin(RELEVANT_STATIONS)]
df_flow["Time"] = pd.to_datetime(df_flow["Timestamp"]).dt.strftime("%H:%M")
df_flow["HOV Flow"] = df_flow["Lane 0 Flow"]
df_flow["HOV Avg Speed"] = df_flow["Lane 0 Avg Speed"]
df_flow["HOV Travel Time"] = df_flow["StationLength"] / df_flow["HOV Avg Speed"]
df_flow["Ordinary Flow"] = 0
df_flow["Ordinary Avg Speed"] = 0
for i in range(1, num_lanes):
    df_flow["Ordinary Flow"] += df_flow[f"Lane {i} Flow"]
    df_flow["Ordinary Avg Speed"] += df_flow[f"Lane {i} Avg Speed"]
df_flow["Ordinary Avg Speed"] /= num_lanes
df_flow["Ordinary Travel Time"] = df_flow["StationLength"] / df_flow["Ordinary Avg Speed"]
df_all = df.merge(df_flow, on = "Time")
df_all_timetoll_gb = df_all[["Time", "Toll", "StationLength", "HOV Travel Time", "Ordinary Travel Time"]].groupby(["Time", "Toll"]).sum().reset_index()
avg_toll = df_all_timetoll_gb["Toll"].mean()
print("Ordinary Travel Time", df_all_timetoll_gb["Ordinary Travel Time"].mean() * 60, "HOV Travel Time", df_all_timetoll_gb["HOV Travel Time"].mean() * 60)
print("Toll:", avg_toll)

## Compute latency in mins, use it to infer \beta
df_all_timetoll_gb["Latency"] = (df_all_timetoll_gb["Ordinary Travel Time"] - df_all_timetoll_gb["HOV Travel Time"]) * 60
beta = (df_all_timetoll_gb["Toll"] / df_all_timetoll_gb["Latency"]).mean()
print("Beta:", beta)

## Compute flows
df_all_station_gb = df_all[["Station", "HOV Flow", "Ordinary Flow"]].groupby("Station").sum().reset_index()
df_all_station_gb["Total Flow"] = df_all_station_gb["HOV Flow"] + df_all_station_gb["Ordinary Flow"]
print("Flow Avg.", df_all_station_gb[["HOV Flow", "Ordinary Flow", "Total Flow"]].mean().reset_index())
print("Flow Std.", df_all_station_gb[["HOV Flow", "Ordinary Flow", "Total Flow"]].std().reset_index())
total_flow = df_all_station_gb["Total Flow"].mean()
ordinary_flow = df_all_station_gb["Ordinary Flow"].mean()
hov_flow = df_all_station_gb["HOV Flow"].mean()

## Calibrate parameters
def solve_gamma2(D, tau, sigma_pool2, sigma_pool3, max_itr = 100, eta = 0.1, eps = 1e-7):
    x = torch.tensor(0.1, requires_grad = True)
    itr = 0
    loss = 1
    loss_arr = []
    while itr < max_itr and loss > eps:
        lhs = x * (1/2 * tau * sigma_pool3 + (3/4 * tau - (tau - x)**2 / tau) * sigma_pool2)
        rhs = 3/8 * tau ** 2 - 1/2 * (tau - x) ** 2
        loss = (lhs - rhs) ** 2
        print(loss)
        loss_arr.append(float(loss.data))
        if loss <= eps:
            break
        loss.backward()
        x.data = x.data - eta * x.grad
        x.grad.zero_()
    x = float(x.detach().numpy())
    return x, loss_arr

sigma_o = hov_flow / total_flow * (num_lanes - 1)
sigma_toll = (1 - sigma_o) / (1 + sigma_2over1 + sigma_3over1)
sigma_pool2 = sigma_toll * sigma_2over1
sigma_pool3 = sigma_toll * sigma_3over1
print("sigma_o:", sigma_o, "sigma_toll", sigma_toll, "sigma_pool2", sigma_pool2, "sigma_pool3:", sigma_pool3)
ugamma3 = 1/2 * avg_toll + 3/4 * sigma_pool2/sigma_pool3
ugamma2 = 3/8 * avg_toll**2 / (sigma_pool3 * ugamma3)
#ugamma2, loss_arr = solve_gamma2(total_flow, avg_toll, sigma_pool2, sigma_pool3, max_itr = 100, eta = 1e-4, eps = 1e-7)
#ugamma3 = 1/2 * avg_toll + (3/4 * avg_toll - (avg_toll - ugamma2)**2 / avg_toll) * sigma_pool2/sigma_pool3
print("ugamma 2:", ugamma2, "ugamma 3:", ugamma3)

#plt.plot(loss_arr)
#plt.title(f"Final Loss: {loss_arr[-1]:.2f}")
#plt.show()

df_latency = df_flow[["Time", "Station", "StationLength", "HOV Travel Time", "Ordinary Travel Time", "HOV Flow", "Ordinary Flow"]].copy()
df_latency["Hour"] = df_flow["Time"].apply(lambda x: x.split(":")[0])
df_latency = df_latency[["Hour", "Station", "StationLength", "HOV Travel Time", "Ordinary Travel Time", "HOV Flow", "Ordinary Flow"]].groupby(["Hour", "Station"]).sum().reset_index()
df_latency_flow = df_latency[["Hour", "HOV Flow", "Ordinary Flow"]].groupby("Hour").mean().reset_index()
df_latency_time = df_latency[["Hour", "StationLength", "HOV Travel Time", "Ordinary Travel Time"]].groupby("Hour").sum().reset_index()
df_latency = df_latency_flow.merge(df_latency_time, on = "Hour")
df_latency["Ordinary Flow"] = df_latency["Ordinary Flow"] / (num_lanes - 1)
df_latency["HOVTimePerMile"] = df_latency["HOV Travel Time"] / df_latency["StationLength"] * 60
df_latency["OrdinaryTimePerMile"] = df_latency["Ordinary Travel Time"] / df_latency["StationLength"] * 60

power = 4

reg = LinearRegression().fit(df_latency[["Ordinary Flow"]] ** power, df_latency["OrdinaryTimePerMile"])
print(reg.coef_, reg.intercept_)
x_lst = np.array(df_latency["Ordinary Flow"])
y_lst = reg.coef_ * x_lst ** power + reg.intercept_
plt.scatter(df_latency["Ordinary Flow"], df_latency["OrdinaryTimePerMile"])
plt.scatter(x_lst, y_lst, color = "red")
#plt.axline((0, reg.intercept_), slope = reg.coef_[0], color = "red")
plt.xlim(df_latency["Ordinary Flow"].min() - 10, df_latency["Ordinary Flow"].max() + 10)
plt.show()

reg = LinearRegression().fit(df_latency[["HOV Flow"]] ** power, df_latency["HOVTimePerMile"])
print(reg.coef_, reg.intercept_)
x_lst = np.array(df_latency["HOV Flow"])
y_lst = reg.coef_ * x_lst ** power + reg.intercept_
plt.scatter(df_latency["HOV Flow"], df_latency["HOVTimePerMile"])
plt.scatter(x_lst, y_lst, color = "red")
#plt.axline((0, reg.intercept_), slope = reg.coef_[0], color = "red")
plt.xlim(df_latency["HOV Flow"].min() - 10, df_latency["HOV Flow"].max() + 10)
plt.show()

X = np.concatenate((np.array(df_latency["Ordinary Flow"]), np.array(df_latency["HOV Flow"])))
X = X.reshape((len(X), 1))
y = np.concatenate((np.array(df_latency["OrdinaryTimePerMile"]), np.array(df_latency["HOVTimePerMile"])))
reg = LinearRegression().fit(X ** power, y)
print(reg.coef_, reg.intercept_)
x_lst = X.reshape((-1,))
y_lst = reg.coef_ * x_lst ** power + reg.intercept_
plt.scatter(X.reshape((-1,)), y)
plt.scatter(x_lst, y_lst, color = "red")
#plt.axline((0, reg.intercept_), slope = reg.coef_[0], color = "red")
plt.xlim(X.min() - 10, X.max() + 10)
plt.show()
