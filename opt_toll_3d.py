import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
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

def tau_larger_than_voftime(tau, rho):
    sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large = get_sigma(tau, 0, rho)
    c_delta = get_travel_time_ordinary(rho, sigma_o) - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3)
#    print(sigma_o, sigma_toll, sigma_pool2, sigma_pool3, c_delta, tau / BETA)
    return c_delta < tau / BETA or not tau_not_too_large

def get_travel_time_ordinary(rho, sigma_o):
    return cost(D * sigma_o / ((1 - rho) * NUM_LANES)) * DISTANCE

def get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3):
    return cost(D * (sigma_toll + 1/2 * sigma_pool2 + 1/3 * sigma_pool3) / (rho * NUM_LANES)) * DISTANCE

def get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3):
    travel_time_o = get_travel_time_ordinary(rho, sigma_o)
    travel_time_hov = get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3)
    return sigma_o * travel_time_o + (sigma_toll + sigma_pool2 + sigma_pool3) * travel_time_hov

def get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3):
    travel_time_o = get_travel_time_ordinary(rho, sigma_o)
    travel_time_hov = get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3)
    return sigma_o * travel_time_o + (sigma_toll + 1/2 * sigma_pool2 + 1/3 * sigma_pool3) * travel_time_hov

def get_total_revenue(tau, sigma_toll):
    return tau * sigma_toll * D

def solve_sigma(tau, rho, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8):
    loss = 1
    loss_arr = []
    if not tau_larger_than_voftime(tau, rho):
        sigma_o, sigma_toll, sigma_pool2, sigma_pool3, _ = get_sigma(tau, 0)
        sigma_toll_init = (1 - sigma_pool2 - sigma_pool3) / 2 #(1 - 1/(UGAMMA2*UGAMMA3) * (1/2*tau*UGAMMA3 + 1/8*tau**2)) / 2 #0.2
        sigma_toll = torch.tensor(sigma_toll_init, requires_grad = True)
        itr = 0
        while loss > eps and itr < max_itr and eta >= min_eta:
            sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large = get_sigma(tau, sigma_toll)
            if not tau_not_too_large:
                loss_arr.append(0)
                break
            loss = torch.abs(get_travel_time_ordinary(rho, sigma_o) / DISTANCE - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3) / DISTANCE - tau / BETA / DISTANCE) ** 2
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
            c_delta = get_travel_time_ordinary(rho, sigma_o) - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3)
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
    
DEBUG = False
RETRAIN = False

## Question: Current pricing scheme minimizes latency because parameters are calibrated from the same dataset?

if DEBUG:
    tau = 15 #4.735
    rho = 3/4
    sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large = solve_sigma(tau, rho, max_itr = 5000, eta = 1e-1, eps = 1e-9, min_eta = 1e-10)
    print(tau_not_too_large)

    print("Sigma_o:", sigma_o, "Sigma_toll:", sigma_toll, "Sigma_pool2:", sigma_pool2, "Sigma_pool3:", sigma_pool3)
    plt.plot(loss_arr)
    plt.title(f"Final Loss: {loss_arr[-1]:.2f}")
    plt.show()
else:
    rho_vals = [1/4, 2/4, 3/4]
    if RETRAIN:
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
        for tau in tqdm(np.arange(0, 15, 0.1)):
            for rho in rho_vals:
                sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large = solve_sigma(tau, rho, max_itr = 5000, eta = 1e-1, eps = 1e-7, min_eta = 1e-10)
                total_travel_time = get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3)
                total_emission = get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3)
                total_revenue = get_total_revenue(tau, sigma_toll)
                
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
        df.to_csv("opt_3d_results.csv", index = False)
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
#    compute_pareto_front(df, "Total Travel Time", "Average Traffic Time Per Traveler (Minutes)", "latency")
#    compute_pareto_front(df, "Total Emission", "Average Emission Per Traveler (Minutes)", "emission")
