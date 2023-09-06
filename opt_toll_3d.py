import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

UGAMMA2 = 28.168
UGAMMA3 = 2.582
BETA = 0.525
POWER = 4
NUM_LANES = 4
a = 3.9427e-12
b = 10.547
D = 5823
DISTANCE = 7.16

def cost(flow):
    return a * flow ** POWER + b

## 1/2 tau < ugamma_3 and tau < ugamma_2
## 1/2 tau < ugamma_3 and tau > ugamma_2 > 1/2 tau
## 1/2 tau < ugamma_3 and 1/2 tau > ugamma_2
## 1/2 tau > ugamma_3 and tau < ugamma_2
## 1/2 tau > ugamma_3 and tau > ugamma_2 > 1/2 tau
## 1/2 tau > ugamma_3 and 1/2 tau > ugamma_2

def get_sigma(tau, sigma_toll):
    const = UGAMMA2 * UGAMMA3
    tau_not_too_large = True
    if 1/2 * tau <= UGAMMA3:
        sigma_pool2 = (1/2 * tau * UGAMMA3 - 1/4 * tau ** 2) / const
    else:
        sigma_pool2 = 0
    if tau <= UGAMMA2:
        sigma_pool3 = (3/4 * tau * min(1/2 * tau, UGAMMA3)) / const
    elif tau > UGAMMA2 and tau / 2 <= UGAMMA2:
        sigma_pool3 = (3/4 * tau * min(1/2 * tau, UGAMMA3) - 1/2 * (tau - UGAMMA2) * min(tau - UGAMMA2, UGAMMA3)) / const
    else:
        sigma_toll = torch.tensor(0.)
        sigma_pool3 = 1/2 * min(tau, UGAMMA3) * UGAMMA2 / const
        tau_not_too_large = False
    sigma_o = 1 - sigma_pool2 - sigma_pool3 - sigma_toll
    return sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large

def sigma_is_feasible(sigma_o, sigma_toll, sigma_pool2, sigma_pool3):
    return sigma_o >= 0 and sigma_o <= 1 and sigma_toll >= 0 and sigma_toll <= 1 and sigma_pool2 >= 0 and sigma_pool2 <= 1 and sigma_pool3 >= 0 and sigma_pool3 <= 1

def tau_larger_than_voftime(tau):
    sigma_o, sigma_toll, sigma_pool2, sigma_pool3, tau_not_too_large = get_sigma(tau, 0)
    c_delta = get_travel_time_ordinary(rho, sigma_o) - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3)
    return c_delta < tau / BETA

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
    if not tau_larger_than_voftime(tau):
        sigma_toll_init = 0.2
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
        sigma_o_init = 0.7
        sigma_o = torch.tensor(sigma_o_init, requires_grad = True)
        itr = 0
        while loss > eps and itr < max_itr and eta >= min_eta:
            sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = sigma_o, 0, 0, 1 - sigma_o
            loss = torch.abs(get_travel_time_ordinary(rho, sigma_o) / DISTANCE - get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3) / DISTANCE - (UGAMMA2 + UGAMMA3) / DISTANCE) ** 2
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

tau = 50 #4.735
rho = 1/4
#sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large = solve_sigma(tau, rho, max_itr = 3000, eta = 1e-1, eps = 1e-7, min_eta = 1e-10)
#
#print("Sigma_o:", sigma_o, "Sigma_toll:", sigma_toll, "Sigma_pool2:", sigma_pool2, "Sigma_pool3:", sigma_pool3)
#plt.plot(loss_arr)
#plt.title(f"Final Loss: {loss_arr[-1]:.2f}")
#plt.show()


total_travel_time_lst = []
total_emission_lst = []
total_revenue_lst = []
loss_lst = []
tau_lst = []
for tau in tqdm(np.arange(0, 60, 0.5)):
    sigma_o, sigma_toll, sigma_pool2, sigma_pool3, loss_arr, tau_not_too_large = solve_sigma(tau, rho, max_itr = 2000, eta = 1e-1, eps = 1e-7, min_eta = 1e-10)
    total_travel_time = get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3)
    total_emission = get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3)
    total_revenue = get_total_revenue(tau, sigma_toll)
    
    total_travel_time_lst.append(total_travel_time)
    total_emission_lst.append(total_emission)
    total_revenue_lst.append(total_revenue)
    loss_lst.append(loss_arr[-1])
    tau_lst.append(tau)

#plt.plot(tau_lst, loss_lst)
#plt.title("Solver Loss")
#plt.show()

plt.plot(tau_lst, total_travel_time_lst, label = "Total Travel Time")
plt.plot(tau_lst, total_emission_lst, label = "Total Emission")
plt.xlabel("Toll Price")
plt.legend()
plt.savefig("3d_latency_vs_toll.png")
plt.clf()
plt.close()

plt.plot(tau_lst, total_revenue_lst, label = "Total Revenue")
plt.xlabel("Toll Price")
plt.legend()
plt.savefig("3d_revenue_vs_toll.png")
plt.clf()
plt.close()

