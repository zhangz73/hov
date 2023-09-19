import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch

#RELEVANT_TIME = ("16:30", "17:30")
RELEVANT_HOUR = 17
RELEVANT_ZONE = "3460 - Hesperian/238 NB"
RELEVANT_STATIONS = [400488, 401561, 400611, 400928, 400284, 400041, 408133, 408135, 417665, 412637, 417666, 408134, 400685, 401003, 400898, 400275, 400939, 400180, 400529, 400990, 400515, 400252]

## Load toll price data
df = pd.read_csv("NB_042021-062021.csv")
#df = df[pd.to_numeric(df["Segment_Toll"], errors='coerce').notnull()]
df = df[pd.to_numeric(df["Zone_Toll"], errors='coerce').notnull()]
df = df.dropna(subset = ["Zone_Toll"])[["dtMsgStartTime2", "siZoneID", "iPlazaID", "Zone_Toll", "Segment_Toll"]]
df = df.fillna(0)
df = df[df["siZoneID"] == RELEVANT_ZONE]
df["Date"] = pd.to_datetime(df["dtMsgStartTime2"]).dt.strftime("%Y-%m-%d")
df["Time"] = pd.to_datetime(df["dtMsgStartTime2"]).dt.strftime("%H:%M")
df = df[df["Date"] == "2021-04-01"]
#df = df[(df["Time"] >= RELEVANT_TIME[0]) & (df["Time"] <= RELEVANT_TIME[1])]
df["Toll"] = df["Zone_Toll"].astype(float)
df = df[["Time", "Toll"]].groupby(["Time"]).mean().reset_index().sort_values("Time")


#plt.plot(pd.to_datetime(df["Time"], format="%H:%M"), df["Toll"])
#plt.gcf().autofmt_xdate()
#plt.show()

## Load population data
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
sigma_2over1 = df_pop["TwoPeople"].sum() * 2 / df_pop["Single"].sum()
sigma_3over1 = df_pop["ThreePlus"].sum() * 3 / df_pop["Single"].sum()
sigma_2over3 = (df_pop["TwoPeople"].sum() * 2) / (df_pop["ThreePlus"].sum() * 3)
sigma_2overhot = df_pop["TwoPeople"].sum() * 2 / (df_pop["Single"].sum() + df_pop["TwoPeople"].sum() * 2 + df_pop["ThreePlus"].sum() * 3)
sigma_3overhot = df_pop["ThreePlus"].sum() * 3 / (df_pop["Single"].sum() + df_pop["TwoPeople"].sum() * 2 + df_pop["ThreePlus"].sum() * 3)

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
## convert per hour speed to per minute speed
df_flow["HOV Avg Speed"] = df_flow["Lane 0 Avg Speed"] / 60
df_flow["HOV Travel Time"] = df_flow["StationLength"] / df_flow["HOV Avg Speed"]
df_flow["Ordinary Flow"] = 0
df_flow["Ordinary Avg Speed"] = 0
for i in range(1, num_lanes):
    df_flow["Ordinary Flow"] += df_flow[f"Lane {i} Flow"]
    df_flow["Ordinary Avg Speed"] += df_flow[f"Lane {i} Avg Speed"] / 60
df_flow["Ordinary Avg Speed"] /= num_lanes
df_flow["Ordinary Travel Time"] = df_flow["StationLength"] / df_flow["Ordinary Avg Speed"]
df_all = df.merge(df_flow, on = "Time")
df_all["Hour"] = df_all["Time"].apply(lambda x: int(x.split(":")[0]))
## Get total toll price, total distance, and total travel time of the selected line segment at each time (5-min)
df_all_timetoll_gb = df_all[["Hour", "Time", "Toll", "StationLength", "HOV Travel Time", "Ordinary Travel Time"]].groupby(["Hour", "Time", "Toll"]).sum().reset_index()
avg_toll = df_all_timetoll_gb["Toll"].mean()
print("Avg Toll:", avg_toll)
## Compute latency in mins, use it to infer \beta
df_all_timetoll_gb["Latency"] = (df_all_timetoll_gb["Ordinary Travel Time"] - df_all_timetoll_gb["HOV Travel Time"])
beta = (df_all_timetoll_gb["Toll"] / df_all_timetoll_gb["Latency"]).mean()
print("Beta:", beta)

df_all_timetoll_gb = df_all_timetoll_gb[df_all_timetoll_gb["Hour"] == RELEVANT_HOUR]
print("Ordinary Travel Time (mins)", df_all_timetoll_gb["Ordinary Travel Time"].mean(), "HOV Travel Time (mins)", df_all_timetoll_gb["HOV Travel Time"].mean())

## Compute per hour flows
df_hour_station_gb = df_all[["Hour", "Station", "HOV Flow", "Ordinary Flow"]].copy()
### Summing up 5-min flows to 1-hr flows
df_hour_station_gb = df_hour_station_gb[["Hour", "Station", "HOV Flow", "Ordinary Flow"]].groupby(["Hour", "Station"]).sum().reset_index()
### Averaging the flows across all stations
df_hour_flow_gb = df_hour_station_gb.groupby("Hour").mean().reset_index()

## Compute per hour avg toll price
df_hour_toll_gb = df_all[["Hour", "Toll"]].groupby("Hour").mean().reset_index()

## Merge to obtain per hour flow & toll prices
df_hour_flow_toll_gb = df_hour_flow_gb.merge(df_hour_toll_gb, on = "Hour")
df_hour_flow_toll_gb = df_hour_flow_toll_gb[(df_hour_flow_toll_gb["Hour"] >= 8) & (df_hour_flow_toll_gb["Hour"] <= 17)]

#plt.plot(df_hour_flow_toll_gb["Hour"], df_hour_flow_toll_gb["HOV Flow"], label = "HOV Flow")
#plt.plot(df_hour_flow_toll_gb["Hour"], df_hour_flow_toll_gb["Ordinary Flow"] / (num_lanes - 1), label = "Ordinary Flow")
#plt.legend()
#plt.title("Flow Per Lane")
#plt.savefig("flow.png")
#plt.clf()
#plt.close()
#
#plt.plot(df_hour_flow_toll_gb["Hour"], df_hour_flow_toll_gb["Toll"])
#plt.show()
#
#assert False

### Compute \ugamma_3 by (\sum_t D_t \sigma_2) / (\sum_t D_t \sigma_3) = r_{2/3}
#ugamma3 = (3 * sigma_2over3 * (df_hour_flow_toll_gb["Toll"] ** 2).sum() + 2 * (df_hour_flow_toll_gb["Toll"] ** 2).sum()) / (4 * df_hour_flow_toll_gb["Toll"]).sum()
#
### Compute \ugamma_2:
####     \sigma_o = f_o / D
####     \sigma_toll + \sigma_2 + \sigma_3 = 1 - \sigma_o
####     \sigma_toll + 1/2\sigma_2 + 1/3\sigma_3 = f_hov / D
#sigma_o_over_hov = (df_hour_flow_toll_gb["Ordinary Flow"] / (num_lanes - 1)) / df_hour_flow_toll_gb["HOV Flow"]
#r_2over3 = (4 * ugamma3 * df_hour_flow_toll_gb["Toll"] - 2 * (df_hour_flow_toll_gb["Toll"] ** 2)) / (3 * (df_hour_flow_toll_gb["Toll"] ** 2))
#toll_coef = 1 + sigma_o_over_hov
#pool3_coef = ((1 + 1/2 * sigma_o_over_hov) / r_2over3 + (1 + 1/3 * sigma_o_over_hov)) * (r_2over3 > 0.05) + (1 + 1/3 * sigma_o_over_hov) * (r_2over3 < 0.05)
#ugamma2 = ((3/8 * df_hour_flow_toll_gb["Toll"] ** 2).sum() / sigma_3over1 + (pool3_coef/toll_coef * 3/8 * df_hour_flow_toll_gb["Toll"] ** 2).sum()) / (ugamma3 / toll_coef).sum()
#print("ugamma 2:", ugamma2, "ugamma 3:", ugamma3)

## Compute \ugamma_2 and \ugamma_3 numerically
def calibrate_ugammas(max_itr = 1000, eps = 1e-7, eta = 1e-2, min_eta = 1e-7):
    ugamma_init = [6.0, 3.0]
    ugammas = torch.tensor(ugamma_init, requires_grad = True)
    loss_arr = []
    loss = 1
    itr = 0
    flow_o = torch.from_numpy(np.array(df_hour_flow_toll_gb["Ordinary Flow"]))
    flow_hov = torch.from_numpy(np.array(df_hour_flow_toll_gb["HOV Flow"]))
    toll = torch.from_numpy(np.array(df_hour_flow_toll_gb["Toll"]))
    while itr < max_itr and loss > eps and eta >= min_eta:
        ## Compute Loss
        D_sigma3 = (flow_o + flow_hov) / (ugammas[0] * ugammas[1] - 1/4 * ugammas[1] * toll - 1/8 * toll ** 2) * (3/8 * toll ** 2)
        D_sigma2 = (flow_o + flow_hov) / (ugammas[0] * ugammas[1] - 1/4 * ugammas[1] * toll - 1/8 * toll ** 2) * (1/2 * ugammas[1] * toll - 1/4 * toll ** 2)
        r2_hot = sigma_2overhot * flow_hov
        r3_hot = sigma_3overhot * flow_hov
        loss = (torch.mean(D_sigma2 - r2_hot)) ** 2 + (torch.mean(D_sigma3 - r3_hot)) ** 2
        loss_arr.append(float(loss.data))
        if loss <= eps:
            break
        if torch.isnan(loss):
            eta = eta * 0.1
            ugammas = torch.tensor(ugamma_init, requires_grad = True)
            loss = 1
            itr = 0
        else:
            loss.backward()
            eta_curr = eta
            ugammas.data = ugammas.data - eta_curr * ugammas.grad
            ugammas.grad.zero_()
        itr += 1
    ugammas = ugammas.detach().numpy()
    return ugammas[0], ugammas[1], loss_arr

ugamma2, ugamma3, loss_arr = calibrate_ugammas(max_itr = 2000, eps = 1e-7, eta = 1e-6, min_eta = 1e-10)
print("ugamma 2:", ugamma2, "ugamma 3:", ugamma3)

plt.plot(loss_arr)
plt.title(f"Final Loss: {loss_arr[-1]:.2f}")
plt.show()

## TODO: Fix it!!!
##  Travel time should NOT be summed across 5-mins, it should be taken average grouping across 5-min intervals
## Travel time can be summed by stations
##  Flows should be summed by 5-mins but averaged across stations
df_latency = df_flow[["Time", "Station", "StationLength", "HOV Travel Time", "Ordinary Travel Time", "HOV Flow", "Ordinary Flow"]].copy()
df_latency["Hour"] = df_flow["Time"].apply(lambda x: x.split(":")[0])
df_latency_flow = df_latency[["Hour", "Station", "StationLength", "HOV Flow", "Ordinary Flow"]].groupby(["Hour", "Station", "StationLength"]).sum().groupby(["Hour"]).mean().reset_index()
df_latency_time = df_latency[["Hour", "Time", "StationLength", "HOV Travel Time", "Ordinary Travel Time"]].groupby(["Hour", "Time"]).sum().groupby(["Hour"]).mean().reset_index()

df_latency = df_latency_flow.merge(df_latency_time, on = "Hour")
df_latency["Ordinary Flow"] = df_latency["Ordinary Flow"] / (num_lanes - 1)
df_latency["HOVTimePerMile"] = df_latency["HOV Travel Time"] / df_latency["StationLength"]
df_latency["OrdinaryTimePerMile"] = df_latency["Ordinary Travel Time"] / df_latency["StationLength"]
print("Total Distance:", df_latency.iloc[0]["StationLength"])
## TODO: Compute total demand per hour using calibrated ugamma2 and ugamma3
df_hour_flow_toll_gb["Total Demand"] = (df_hour_flow_toll_gb["Ordinary Flow"] + df_hour_flow_toll_gb["HOV Flow"]) / (ugamma2 * ugamma3 - 1/4 * ugamma3 * df_hour_flow_toll_gb["Toll"] - 1/8 * df_hour_flow_toll_gb["Toll"] ** 2) * ugamma2 * ugamma3
print("Total Demand:", df_hour_flow_toll_gb[df_hour_flow_toll_gb["Hour"] == RELEVANT_HOUR].iloc[0]["Total Demand"])
df_hour_flow_toll_gb.to_csv("hourly_demand_20210401.csv", index = False)
#print("Total Demand:", df_latency[df_latency["Hour"] == RELEVANT_HOUR].iloc[0]["Total Demand"])

power = 4

#reg = LinearRegression().fit(df_latency[["Ordinary Flow"]] ** power, df_latency["OrdinaryTimePerMile"])
#print(reg.coef_, reg.intercept_)
#x_lst = np.array(df_latency["Ordinary Flow"])
#y_lst = reg.coef_ * x_lst ** power + reg.intercept_
#plt.scatter(df_latency["Ordinary Flow"], df_latency["OrdinaryTimePerMile"])
#plt.scatter(x_lst, y_lst, color = "red")
##plt.axline((0, reg.intercept_), slope = reg.coef_[0], color = "red")
#plt.xlim(df_latency["Ordinary Flow"].min() - 10, df_latency["Ordinary Flow"].max() + 10)
#plt.show()
#
#reg = LinearRegression().fit(df_latency[["HOV Flow"]] ** power, df_latency["HOVTimePerMile"])
#print(reg.coef_, reg.intercept_)
#x_lst = np.array(df_latency["HOV Flow"])
#y_lst = reg.coef_ * x_lst ** power + reg.intercept_
#plt.scatter(df_latency["HOV Flow"], df_latency["HOVTimePerMile"])
#plt.scatter(x_lst, y_lst, color = "red")
##plt.axline((0, reg.intercept_), slope = reg.coef_[0], color = "red")
#plt.xlim(df_latency["HOV Flow"].min() - 10, df_latency["HOV Flow"].max() + 10)
#plt.show()

X = np.concatenate((np.array(df_latency["Ordinary Flow"]), np.array(df_latency["HOV Flow"])))
X = X.reshape((len(X), 1))
y = np.concatenate((np.array(df_latency["OrdinaryTimePerMile"]), np.array(df_latency["HOVTimePerMile"])))
reg = LinearRegression().fit(X ** power, y)
print(reg.coef_, reg.intercept_)
x_lst = X.reshape((-1,))
y_lst = reg.coef_ * x_lst ** power + reg.intercept_
y_lst2 = (0.15 / 140) ** 4 * x_lst ** power + reg.intercept_
plt.scatter(X.reshape((-1,)), y)
plt.scatter(x_lst, y_lst, color = "red")
#plt.axline((0, reg.intercept_), slope = reg.coef_[0], color = "red")
plt.xlim(X.min() - 10, X.max() + 10)
plt.show()

print(reg.coef_ ** 0.25)
#print(0.15 / (reg.coef_ ** 0.25))
