import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

UGAMMA2 = 13.517 #5.024 #19.536 $
UGAMMA3 = 2.711 #2.643 #2.510 $
BETA = 0.376 #0.525 $/min
POWER = 4
NUM_LANES = 4
a = 3.2856e-13 #3.9427e-12
b = 0.8789 #10.547
D = 6522 #5363 # people/hr
DISTANCE = 7.16 # miles
## Ordinary Lane Travel Time at 17 pm: 18.28 mins
## HOT Lane Travel Time at 17 pm: 9.46 mins

TIME_FREQ = 5
RELEVANT_ZONE = "3460 - Hesperian/238 NB"
df_demand = pd.read_csv("hourly_demand_20210401.csv")
df_demand = df_demand[["Hour", "Total Demand"]]
df_toll = pd.read_csv("NB_042021-062021.csv")
df_toll = df_toll[pd.to_numeric(df_toll["Zone_Toll"], errors='coerce').notnull()]
df_toll = df_toll.dropna(subset = ["Zone_Toll"])[["dtMsgStartTime2", "siZoneID", "iPlazaID", "Zone_Toll", "Segment_Toll"]]
df_toll = df_toll.fillna(0)
df_toll = df_toll[df_toll["siZoneID"] == RELEVANT_ZONE]
df_toll["Date"] = pd.to_datetime(df_toll["dtMsgStartTime2"]).dt.strftime("%Y-%m-%d")
df_toll["Time"] = pd.to_datetime(df_toll["dtMsgStartTime2"]).dt.strftime("%H:%M")
df_toll["Hour"] = df_toll["Time"].apply(lambda x: int(x.split(":")[0]))
df_toll = df_toll[df_toll["Date"] == "2021-04-01"]
df_toll["Toll"] = df_toll["Zone_Toll"].astype(float)
df_all = df_demand.merge(df_toll, on = "Hour")
df_all = df_all[["Time", "Hour", "Total Demand", "Toll"]].drop_duplicates().sort_values("Time").reset_index()

def cost(flow):
    return a * flow ** POWER + b

def auc_rec(x_max, y_max, target):
    ## Assume x_max >= y_max
    if x_max < y_max:
        return auc_rec(y_max, x_max, target)
    const = x_max * y_max
    if target <= y_max:
        return 1/2 * target ** 2 / const
    elif target <= x_max:
        return 1/2 * (target + target - y_max) * y_max / const
    else:
        return 1 - 1/2 * (x_max + y_max - target) ** 2 / const

## 1/2 tau < ugamma_3 and tau < ugamma_2
## 1/2 tau < ugamma_3 and tau > ugamma_2 > 1/2 tau
## 1/2 tau < ugamma_3 and 1/2 tau > ugamma_2
## 1/2 tau > ugamma_3 and tau < ugamma_2
## 1/2 tau > ugamma_3 and tau > ugamma_2 > 1/2 tau
## 1/2 tau > ugamma_3 and 1/2 tau > ugamma_2
## When \beta C_delta = \tau holds, people who prefer carpool than tolling will NOT take ordinary lane
## When sigma_toll = 0, there are 2 cases:
##   1) When 1/2 tau > ugamma3: People split between carpool 3 and ordinary lane
##   2) When 1/2 tau <= ugamma3: TBD
def get_sigma(tau, sigma_toll, rho = 0):
    const = UGAMMA2 * UGAMMA3
    tau_not_too_large = True
    if 1/2 * tau <= UGAMMA3:
        sigma_pool2 = 1/2 * (UGAMMA3 - 1/2*tau) * min(1/2*tau, UGAMMA2) / const
    else:
        sigma_pool2 = 0
    if tau <= UGAMMA2:
        if 1/2 * tau <= UGAMMA3:
            sigma_pool3 = (3/4 * tau * 1/2 * tau) / const
        else:
            sigma_pool3 = 1/2 * (2*tau - UGAMMA3) * UGAMMA3 / const
    elif tau > UGAMMA2 and tau / 2 <= UGAMMA2:
        if tau - UGAMMA2 <= UGAMMA3:
            sigma_pool3 = (3/4 * tau * min(1/2 * tau, UGAMMA3) - 1/2 * (tau - UGAMMA2) * min(tau - UGAMMA2, UGAMMA3)) / const
        else:
            sigma_pool3 = (3/4 * tau * min(1/2 * tau, UGAMMA3) - 1/2 * (2*tau - 2*UGAMMA2 - UGAMMA3)*UGAMMA3) / const
    else:
        sigma_toll = torch.tensor(0.)
        ## Placeholder
        sigma_pool3 = 1/2 #1/2 * min(tau, UGAMMA3) * UGAMMA2 / const
        tau_not_too_large = False
    sigma_o = 1 - sigma_pool2 - sigma_pool3 - sigma_toll
    return sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large

def sigma_is_feasible(sigma_o, sigma_toll, sigma_pool2, sigma_pool3):
    return sigma_o >= 0 and sigma_o <= 1 and sigma_toll >= 0 and sigma_toll <= 1 and sigma_pool2 >= 0 and sigma_pool2 <= 1 and sigma_pool3 >= 0 and sigma_pool3 <= 1

def tau_larger_than_voftime(tau, rho, D = D):
    sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large = get_sigma(tau, 0, rho)
    c_delta = get_travel_time_ordinary(rho, sigma_o, D = D) - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D = D)
#    print(sigma_o, sigma_toll, sigma_pool2, sigma_pool3, c_delta, tau / BETA)
    return c_delta < tau / BETA or not tau_not_too_large

def get_travel_time_ordinary(rho, sigma_o, D = D):
    return cost(D * sigma_o / ((1 - rho) * NUM_LANES)) * DISTANCE

def get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D = D):
    return cost(D * (sigma_toll + 1/2 * sigma_pool2 + 1/3 * sigma_pool3) / (rho * NUM_LANES)) * DISTANCE

def get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D):
    travel_time_o = get_travel_time_ordinary(rho, sigma_o, D = D)
    travel_time_hov = get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D = D)
    return sigma_o * travel_time_o + (sigma_toll + sigma_pool2 + sigma_pool3) * travel_time_hov

def get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D):
    travel_time_o = get_travel_time_ordinary(rho, sigma_o, D = D)
    travel_time_hov = get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D = D)
    return sigma_o * travel_time_o + (sigma_toll + 1/2 * sigma_pool2 + 1/3 * sigma_pool3) * travel_time_hov

def get_total_revenue(tau, sigma_toll, D = D):
    return tau * sigma_toll * D

def solve_sigma(tau, rho, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8, D = D):
    loss = 1
    loss_arr = []
    if not tau_larger_than_voftime(tau, rho, D = D):
        sigma_o, sigma_toll, sigma_pool2, sigma_pool3, _ = get_sigma(tau, 0)
        sigma_toll_init = (1 - sigma_pool2 - sigma_pool3) / 2 #(1 - 1/(UGAMMA2*UGAMMA3) * (1/2*tau*UGAMMA3 + 1/8*tau**2)) / 2 #0.2
        sigma_toll = torch.tensor(sigma_toll_init, requires_grad = True)
        itr = 0
        while loss > eps and itr < max_itr and eta >= min_eta:
            sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large = get_sigma(tau, sigma_toll)
            if not tau_not_too_large:
                loss_arr.append(0)
                break
            loss = torch.abs(get_travel_time_ordinary(rho, sigma_o, D = D) / DISTANCE - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D = D) / DISTANCE - tau / BETA / DISTANCE) ** 2
            loss_arr.append(float(loss.data))
            rerun = False
            if loss <= eps:
                break
            if torch.isnan(loss) or itr >= max_itr - 1:
                itr = 0
                eta /= 10
                loss_arr = []
                sigma_toll = torch.tensor(sigma_toll_init, requires_grad = True)
                rerun = True
                loss = 1
            if not rerun:
                loss.backward()
                sigma_toll.data = sigma_toll.data - eta * sigma_toll.grad
                sigma_toll.grad.zero_()
                itr += 1
        sigma_toll = float(sigma_toll.detach().data)
        sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large = get_sigma(tau, sigma_toll)
    else:
        sigma_o_init = (1-rho)/(1+2*rho) #(1 + (1-rho)/(1+2*rho)) / 2 #0.5
        sigma_o = torch.tensor(sigma_o_init, requires_grad = True)
        itr = 0
        while loss > eps and itr < max_itr and eta >= min_eta:
            sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = sigma_o, 0, 0, 1 - sigma_o
            c_delta = get_travel_time_ordinary(rho, sigma_o, D = D) - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D = D)
            target_sigma_pool3 = auc_rec(UGAMMA2, UGAMMA3, BETA * c_delta)
#            print(BETA * c_delta, target_sigma_pool3)
            loss = torch.abs(target_sigma_pool3 - sigma_pool3) ** 2
            loss_arr.append(float(loss.data))
            rerun = False
            if loss <= eps:
                break
            if torch.isnan(loss) or itr >= max_itr - 1:
                itr = 0
                eta /= 10
                loss_arr = []
                sigma_o = torch.tensor(sigma_o_init, requires_grad = True)
                rerun = True
                loss = 1
            if not rerun:
                loss.backward()
                sigma_o.data = sigma_o.data - eta * sigma_o.grad
                sigma_o.grad.zero_()
                itr += 1
        sigma_o = float(sigma_o.detach().data)
        sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = sigma_o, 0, 0, 1 - sigma_o
        tau_not_too_large = False
    return sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large

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

def grid_search(rho_vals = [1/4, 2/4, 3/4], toll_lst = [], save_to_file = True, D = D):
    total_travel_time_lst = []
    total_emission_lst = []
    total_revenue_lst = []
    loss_lst = []
    tau_lst = []
    rho_lst = []
    sigma_o_lst = []
    sigma_toll_lst = []
    sigma_pool2_lst = []
    sigma_pool3_lst = []
    for tau in tqdm(toll_lst, leave = False):
        for rho in rho_vals:
            sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large = solve_sigma(tau, rho, max_itr = 5000, eta = 1e-1, eps = 1e-7, min_eta = 1e-10, D = D)
            total_travel_time = get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
            total_emission = get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
            total_revenue = get_total_revenue(tau, sigma_toll, D = D)
            
            if len(loss_arr) == 0:
#                print(tau, rho)
                loss_arr = [0.]
            
            total_travel_time_lst.append(total_travel_time)
            total_emission_lst.append(total_emission)
            total_revenue_lst.append(total_revenue)
            loss_lst.append(loss_arr[-1])
            tau_lst.append(tau)
            rho_lst.append(rho)
            sigma_o_lst.append(sigma_o)
            sigma_toll_lst.append(sigma_toll)
            sigma_pool2_lst.append(sigma_pool2)
            sigma_pool3_lst.append(sigma_pool3)

    df = pd.DataFrame.from_dict({"% Ordinary": sigma_o_lst, "% Toll": sigma_toll_lst, "% Pool 2": sigma_pool2_lst, "% Pool 3": sigma_pool3_lst, "Total Travel Time": total_travel_time_lst, "Total Emission": total_emission_lst, "Total Revenue": total_revenue_lst, "Loss": loss_lst, "Toll Price": tau_lst, "HOT Capacity": rho_lst})
    if save_to_file:
        df.to_csv("opt_3d_results.csv", index = False)
    return df

"""
### Pipeline for 5-6 pm optimal toll & HOT capacity design
DEBUG = False
RETRAIN = False

## Question: Current pricing scheme minimizes latency because parameters are calibrated from the same dataset?

if DEBUG:
    tau = 15 #4.735
    rho = 3/4
    sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large = solve_sigma(tau, rho, max_itr = 5000, eta = 1e-1, eps = 1e-9, min_eta = 1e-10, D = D)
    print(tau_not_too_large)

    print("Sigma_o:", sigma_o, "Sigma_toll:", sigma_toll, "Sigma_pool2:", sigma_pool2, "Sigma_pool3:", sigma_pool3)
    plt.plot(loss_arr)
    plt.title(f"Final Loss: {loss_arr[-1]:.2f}")
    plt.show()
else:
    rho_vals = [1/4, 2/4, 3/4]
    if RETRAIN:
        df = grid_search(rho_vals = rho_vals, tau_lst = np.arange(0, 15, 0.1), save_to_file = True, D = D)
    else:
        df = pd.read_csv("opt_3d_results.csv")

    #plt.plot(tau_lst, loss_lst)
    #plt.title("Solver Loss")
    #plt.show()

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
    
    ## Compute Pareto Fronts
    compute_pareto_front(df, "Total Travel Time", "Average Traffic Time Per Traveler (Minutes)", "latency")
    compute_pareto_front(df, "Total Emission", "Average Emission Per Traveler (Minutes)", "emission")
"""

### Pipeline for dynamic toll & HOT capacity design
RETRAIN = False
if RETRAIN:
    rho_vals = [1/4, 2/4, 3/4]
    hour_lst = []
    rho_lst = []
    min_congestion_lst = []
    min_congestion_tau_lst = []
    min_emission_lst = []
    min_emission_tau_lst = []
    max_revenue_lst = []
    max_revenue_tau_lst = []
    for t in tqdm(np.array(df_all["Hour"].unique())):
        demand = df_all[df_all["Hour"] == t].iloc[0]["Total Demand"]
        df = grid_search(rho_vals = rho_vals, toll_lst = np.arange(0, 15, 0.1), save_to_file = False, D = demand)
        for rho in tqdm(rho_vals, leave = False):
            df_tmp = df[df["HOT Capacity"] == rho]
            min_congestion = df_tmp["Total Travel Time"].min()
            min_congestion_tau = df_tmp.iloc[df_tmp["Total Travel Time"].argmin()]["Toll Price"]
            min_emission = df_tmp["Total Emission"].min()
            min_emission_tau = df_tmp.iloc[df_tmp["Total Emission"].argmin()]["Toll Price"]
            max_revenue = df_tmp["Total Revenue"].min()
            max_revenue_tau = df_tmp.iloc[df_tmp["Total Revenue"].argmax()]["Toll Price"]
            hour_lst.append(t)
            rho_lst.append(rho)
            min_congestion_lst.append(min_congestion)
            min_congestion_tau_lst.append(min_congestion_tau)
            min_emission_lst.append(min_emission)
            min_emission_tau_lst.append(min_emission_tau)
            max_revenue_lst.append(max_revenue)
            max_revenue_tau_lst.append(max_revenue_tau)

    df_dynamic = pd.DataFrame.from_dict({"Hour": hour_lst, "Rho": rho_lst, "Min Congestion": min_congestion_lst, "Min Congestion Toll": min_congestion_tau_lst, "Min Emission": min_emission_lst, "Min Emission Toll": min_emission_tau_lst, "Max Revenue": max_revenue_lst, "Max Revenue Toll": max_revenue_tau_lst})
    df_dynamic.to_csv("time_dynamic_design.csv", index = False)
else:
    df_dynamic = pd.read_csv("time_dynamic_design.csv")

feat_lst = ["Min Congestion", "Min Emission", "Max Revenue"]
rho = 0.25
for feat in feat_lst:
    time_lst = pd.to_datetime(df_all["Time"], format="%H:%M")
    opt_toll = np.repeat(np.array(df_dynamic[df_dynamic["Rho"] == rho][f"{feat} Toll"]), 60 // TIME_FREQ)
    curr_toll = df_all["Toll"]
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
