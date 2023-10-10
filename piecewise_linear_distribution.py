import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

## Load Data
df = pd.read_csv("hourly_demand_20210401.csv")
TAU_LST = list(df["Toll"])
LATENCY_O_LST = list(df["Ordinary Travel Time"])
LATENCY_HOV_LST = list(df["HOV Travel Time"])
FLOW_O_LST = list(df["Ordinary Flow"])
FLOW_HOV_LST = list(df["HOV Flow"])

## Hyperparameters
VOT_CHUNKS = 2
CARPOOL2_CHUNKS = 2
CARPOOL3_CHUNKS = 2
BETA_RANGE = (0, 1)
GAMMA2_RANGE = (0, 20)
GAMMA3_RANGE = (0, 5)
INT_GRID = 50

## Latency Parameters
POWER = 4
NUM_LANES = 4
a = 3.2856e-13 #3.9427e-12
b = 0.8789 #10.547
DISTANCE = 7.16 # miles

## Helper functions
def cost(flow):
    return a * flow ** POWER + b
    
def get_travel_time_ordinary(rho, sigma_o, D):
    return cost(D * sigma_o / ((1 - rho) * NUM_LANES)) * DISTANCE

def get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D):
    return cost(D * (sigma_toll + 1/2 * sigma_pool2 + 1/3 * sigma_pool3) / (rho * NUM_LANES)) * DISTANCE

def get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D):
    travel_time_o = get_travel_time_ordinary(rho, sigma_o, D)
    travel_time_hov = get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D)
    return sigma_o * travel_time_o + (sigma_toll + sigma_pool2 + sigma_pool3) * travel_time_hov

def get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D):
    travel_time_o = get_travel_time_ordinary(rho, sigma_o, D)
    travel_time_hov = get_travel_time_hov(rho, sigma_toll, sigma_pool2, sigma_pool3, D)
    return sigma_o * travel_time_o + (sigma_toll + 1/2 * sigma_pool2 + 1/3 * sigma_pool3) * travel_time_hov

def get_total_revenue(tau, sigma_toll, D):
    return tau * sigma_toll * D

## Calibrate preference distribution
def get_atomic_strategy_profile(beta, gamma2, gamma3, tau, latency_o, latency_hov, is_scalar = False):
    cost_o = beta * latency_o
    cost_toll = beta * latency_hov + tau
    cost_pool2 = beta * latency_hov + 0.5 * tau + gamma2
    cost_pool3 = beta * latency_hov + gamma2 + gamma3
    if is_scalar:
        cost_arr = torch.tensor([cost_o, cost_toll, cost_pool2, cost_pool3])
        return torch.argmin(cost_arr)
    strategy_profile = torch.zeros((4, len(cost_o)))
    strategy_profile[0,:] += (cost_o <= cost_toll) * (cost_o <= cost_pool2) * (cost_o <= cost_pool3)
    strategy_profile[1,:] += (cost_toll <= cost_o) * (cost_toll <= cost_pool2) * (cost_toll <= cost_pool3)
    strategy_profile[2,:] += (cost_pool2 <= cost_o) * (cost_pool2 <= cost_toll) * (cost_pool2 <= cost_pool3)
    strategy_profile[3,:] += (cost_pool3 <= cost_o) * (cost_pool3 <= cost_toll) * (cost_pool3 <= cost_pool2)
    strategy_profile = strategy_profile / torch.sum(strategy_profile, axis = 0)
    return strategy_profile

def get_cube_strategy_profile(beta_lo, beta_hi, gamma2_lo, gamma2_hi, gamma3_lo, gamma3_hi, tau, latency_o, latency_hov):
    c_delta = latency_o - latency_hov
    ## Toll: beta * c_delta > tau, gamma2 > 0.5 tau, gamma2 + gamma3 > tau
    if beta_lo < tau / c_delta or gamma2_hi < 0.5 * tau:
        sigma_toll = 0
    else:
        beta_frac = (beta_hi - tau / c_delta) / (beta_hi - beta_lo)
        ### \int_{max(0.5*tau, \gamma2_lo)}^{\gamma2_hi} \int_{max(\tau-\gamma_2, \gamma3_lo)}^{\gamma3_hi} 1/(gamma2_len*gamma3_len) d\gamma3 d\gamma_2
        gamma2_vec = np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1)
        gamma3_vec = np.linspace(gamma3_lo, gamma3_hi, INT_GRID + 1)
        gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
        gamma3_vec = (gamma3_vec[1:] + gamma3_vec[:-1]) / 2
        gamma23_mat = np.zeros((INT_GRID, INT_GRID))
        for i in range(INT_GRID):
            gamma2_val = gamma2_vec[i]
            if gamma2_val >= 0.5 * tau:
                lo = max(tau - gamma2_val, gamma2_lo)
                gamma23_mat[i,:] += (gamma3_vec >= lo)
        gamma23_frac = np.mean(gamma23_mat)
        sigma_toll = beta_frac * gamma23_frac
    ## Pool2: beta * c_delta > 0.5 tau + gamma2, gamma2 < 0.5 tau, gamma3 > 0.5 tau
    if gamma2_lo > 0.5 * tau or gamma3_hi < 0.5 * tau:
        sigma_pool2 = 0
    else:
        gamma3_frac = (gamma3_hi - 0.5 * tau) / (gamma3_hi - gamma3_lo)
        ### \int_{\gamma2_lo}^{min(0.5*tau, \gamma2_hi)} \int_{max(\beta_lo, (0.5*tau+\gamma_2)/c_delta)}^{\beta_hi}
        gamma2_vec = np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1)
        beta_vec = np.linspace(beta_lo, beta_hi, INT_GRID + 1)
        gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
        beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
        gamma2beta_mat = np.zeros((INT_GRID, INT_GRID))
        for i in range(INT_GRID):
            gamma2_val = gamma2_vec[i]
            if gamma2_val <= 0.5 * tau:
                lo = max((0.5*tau+gamma2_val)/c_delta, beta_lo)
                gamma2beta_mat[i,:] += (beta_vec >= lo)
        gamma2beta_frac = np.mean(gamma2beta_mat)
        sigma_pool2 = gamma3_frac * gamma2beta_frac
    ## Pool3: beta * c_delta > gamma2 + gamma3, gamma3 < 0.5 tau, gamma2 + gamma3 < tau
    gamma2_vec = np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1)
    gamma3_vec = np.linspace(gamma3_lo, gamma3_hi, INT_GRID + 1)
    beta_vec = np.linspace(beta_lo, beta_hi, INT_GRID + 1)
    gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
    gamma3_vec = (gamma3_vec[1:] + gamma3_vec[:-1]) / 2
    beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
    gamma32beta_mat = np.zeros((INT_GRID, INT_GRID, INT_GRID))
    for i in range(INT_GRID):
        gamma3_val = gamma3_vec[i]
        if gamma3_val <= 0.5 * tau:
            for j in range(INT_GRID):
                gamma2_val = gamma2_vec[j]
                if gamma2_val + gamma3_val <= tau:
                    lo = (gamma2_val + gamma3_val) / c_delta
                    gamma32beta_mat[i,j,:] += (beta_vec >= lo)
    sigma_pool3 = np.mean(gamma32beta_mat)
    ## Ordinary lane: beta * c_delta < min(tau, gamma2, gamma3)
    sigma_o = 1 - sigma_toll - sigma_pool2 - sigma_pool3
    return sigma_o, sigma_toll, sigma_pool2, sigma_pool3

def get_entire_strategy_profile(tau, latency_o, latency_hov):
    beta_vec = np.linspace(BETA_RANGE[0], BETA_RANGE[1], VOT_CHUNKS + 1)
    gamma2_vec = np.linspace(GAMMA2_RANGE[0], GAMMA2_RANGE[1], CARPOOL2_CHUNKS + 1)
    gamma3_vec = np.linspace(GAMMA3_RANGE[0], GAMMA3_RANGE[1], CARPOOL3_CHUNKS + 1)
    sigma_profile = np.zeros((VOT_CHUNKS * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS, 4))
    for beta_idx in range(VOT_CHUNKS):
        for gamma2_idx in range(CARPOOL2_CHUNKS):
            for gamma3_idx in range(CARPOOL3_CHUNKS):
                beta_lo, beta_hi = beta_vec[beta_idx], beta_vec[beta_idx + 1]
                gamma2_lo, gamma2_hi = gamma2_vec[gamma2_idx], gamma2_vec[gamma2_idx + 1]
                gamma3_lo, gamma3_hi = gamma3_vec[gamma3_idx], gamma3_vec[gamma3_idx + 1]
                sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = get_cube_strategy_profile(beta_lo, beta_hi, gamma2_lo, gamma2_hi, gamma3_lo, gamma3_hi, tau, latency_o, latency_hov)
                loc = beta_idx * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS + gamma2_idx * CARPOOL3_CHUNKS + gamma3_idx
                sigma_profile[loc, 0] = sigma_o
                sigma_profile[loc, 1] = sigma_toll
                sigma_profile[loc, 2] = sigma_pool2
                sigma_profile[loc, 3] = sigma_pool3
    return sigma_profile

def calibrate_density(tau_lst, latency_o_lst, latency_hov_lst, flow_o_lst, flow_hov_lst, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8):
    assert len(tau_lst) == len(flow_o_lst) and len(tau_lst) == len(flow_hov_lst) and len(tau_lst) == len(latency_o_lst) and len(tau_lst) == len(latency_hov_lst)
    ## Step 1: Construct sigma profiles
    data_size = len(tau_lst)
    profile_len = VOT_CHUNKS * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS
    sigma_profile_lst = np.zeros((data_size, profile_len, 4))
    for i in tqdm(range(data_size)):
        sigma_profile_lst[i,:,:] = get_entire_strategy_profile(tau_lst[i], latency_o_lst[i], latency_hov_lst[i])
    ## Step 2: Compute density
    tau_lst = torch.tensor(tau_lst)
    flow_o_lst = torch.tensor(flow_o_lst)
    flow_hov_lst = torch.tensor(flow_hov_lst)
    sigma_profile_lst = torch.from_numpy(sigma_profile_lst)
    coef_mat = torch.ones((data_size, 4))
    coef_mat[:,0] = -flow_hov_lst
    coef_mat[:,1] = flow_o_lst
    coef_mat[:,2] = 1/2 * flow_o_lst
    coef_mat[:,3] = 1/3 * flow_o_lst
    ### Iterate
    f = torch.ones(profile_len, requires_grad = True)
    loss_diff = 1
    loss = 1
    itr = 0
    loss_arr = []
    while loss_diff > eps and itr < max_itr and eta >= min_eta:
        entire_profiles = sigma_profile_lst.permute(0, 2, 1) * torch.abs(f) #*torch.arange(sigma_profile_lst.ndim - 1, -1, -1)
        entire_profiles_perm = entire_profiles.permute(0, 2, 1)
        entire_profiles_sum = torch.sum(entire_profiles_perm, axis = 1)
        loss = torch.mean(torch.abs(torch.sum(coef_mat * entire_profiles_sum, axis = 1)) ** 2)
        loss_arr.append(float(loss.data))
        rerun = False
        if len(loss_arr) <= 1:
            loss_diff = 1
        else:
            loss_diff = abs(loss_arr[-1] - loss_arr[-2])
        if loss_diff <= eps:
            break
        if torch.isnan(loss) or itr >= max_itr - 1:
            itr = 0
            eta /= 10
            loss_arr = []
            f = torch.ones(profile_len, requires_grad = True)
            rerun = True
            loss = 1
            loss_diff = 1
        if not rerun:
            loss.backward()
            f.data = f.data - eta * f.grad
            f.grad.zero_()
            itr += 1
    f = torch.abs(f) / torch.sum(torch.abs(f))
    return f.detach().numpy(), loss_arr

## Discretize calibrated distribution into fine-grained cubes
def get_preference_cubes(preference_density):
    beta_lst = np.linspace(BETA_RANGE[0], BETA_RANGE[1], INT_GRID + 1)
    gamma2_lst = np.linspace(GAMMA2_RANGE[0], GAMMA2_RANGE[1], INT_GRID + 1)
    gamma3_lst = np.linspace(GAMMA3_RANGE[0], GAMMA3_RANGE[1], INT_GRID + 1)
    beta_val_lst = np.linspace(BETA_RANGE[0], BETA_RANGE[1], VOT_CHUNKS + 1)
    gamma2_val_lst = np.linspace(GAMMA2_RANGE[0], GAMMA2_RANGE[1], CARPOOL2_CHUNKS + 1)
    gamma3_val_lst = np.linspace(GAMMA3_RANGE[0], GAMMA3_RANGE[1], CARPOOL3_CHUNKS + 1)
        
    preference_len = INT_GRID ** 3
    beta_vec = np.zeros(preference_len)
    gamma2_vec = np.zeros(preference_len)
    gamma3_vec = np.zeros(preference_len)
    density_vec = np.zeros(preference_len)
    curr_beta_idx = 0
    for beta_idx in range(INT_GRID):
        curr_gamma2_idx = 0
        while beta_lst[beta_idx] > beta_val_lst[curr_beta_idx + 1]:
            curr_beta_idx += 1
        for gamma2_idx in range(INT_GRID):
            curr_gamma3_idx = 0
            while gamma2_lst[gamma2_idx] > gamma2_val_lst[curr_gamma2_idx + 1]:
                curr_gamma2_idx += 1
            for gamma3_idx in range(INT_GRID):
                while gamma3_lst[gamma3_idx] > gamma3_val_lst[curr_gamma3_idx + 1]:
                    curr_gamma3_idx += 1
                loc = curr_beta_idx * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS + curr_gamma2_idx * CARPOOL3_CHUNKS + curr_gamma3_idx
                f = preference_density[loc]
                idx = beta_idx * INT_GRID ** 2 + gamma2_idx * INT_GRID + gamma3_idx
                beta_vec[idx] = beta_lst[beta_idx]
                gamma2_vec[idx] = gamma2_lst[gamma2_idx]
                gamma3_vec[idx] = gamma3_lst[gamma3_idx]
                density_vec[idx] = f
    density_vec = density_vec / np.sum(density_vec)
    return beta_vec, gamma2_vec, gamma3_vec, density_vec

## Optimal toll & capacity design
def get_equilibrium_profile(beta_vec, gamma2_vec, gamma3_vec, density_vec, tau, rho, D, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8):
    beta_vec = torch.from_numpy(beta_vec)
    gamma2_vec = torch.from_numpy(gamma2_vec)
    gamma3_vec = torch.from_numpy(gamma3_vec)
    density_vec = torch.from_numpy(density_vec)
    
    sigma_target_init = [0.25, 0.25, 0.25, 0.25]
    sigma_target = torch.tensor(sigma_target_init, requires_grad = True)
    loss = 1
    itr = 0
    loss_arr = []
    sigma_opt = sigma_target.clone().detach()
    loss_opt = 1
    while loss > eps and itr < max_itr and eta >= min_eta:
        latency_o = get_travel_time_ordinary(rho, sigma_target[0], D)
        latency_hov = get_travel_time_hov(rho, sigma_target[1], sigma_target[2], sigma_target[3], D)
        sigma_output_vec = get_atomic_strategy_profile(beta_vec, gamma2_vec, gamma3_vec, tau, latency_o, latency_hov, is_scalar = False)
        sigma_output = torch.sum(sigma_output_vec * density_vec, axis = 1)
        loss = torch.sum((sigma_target - sigma_output) ** 2)
        loss_arr.append(float(loss.data))
        if loss < loss_opt:
            loss_opt = float(loss.data)
            sigma_opt = sigma_target.clone().detach()
        rerun = False
        if loss <= eps:
            break
        if torch.isnan(loss) or itr >= max_itr - 1:
            itr = 0
            eta /= 10
            loss_arr = []
            sigma_target = torch.tensor(sigma_target_init, requires_grad = True)
            rerun = True
            loss = 1
        if not rerun:
            loss.backward()
            sigma_target.data = sigma_target.data - eta * sigma_target.grad
            sigma_target.grad.zero_()
#            sigma_target.data = (sigma_target.data + sigma_output.data) / 2
            itr += 1
#    print(loss_opt)
    return sigma_opt.numpy(), loss_arr

preference_density, loss_arr = calibrate_density(TAU_LST, LATENCY_O_LST, LATENCY_HOV_LST, FLOW_O_LST, FLOW_HOV_LST, max_itr = 20000, eta = 1e-1, eps = 1e-5, min_eta = 1e-8)
print([round(x, 2) for x in preference_density])
#plt.plot(loss_arr)
#plt.title(f"Final Loss = {loss_arr[-1]:.2e}")
#plt.show()

beta_vec, gamma2_vec, gamma3_vec, density_vec = get_preference_cubes(preference_density)

tau = 0.3
rho = 0.25
D = 6500

sigma, loss_arr = get_equilibrium_profile(beta_vec, gamma2_vec, gamma3_vec, density_vec, tau, rho, D, max_itr = 2000, eta = 1e-1, eps = 2e-3, min_eta = 1e-8)
sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = sigma[0], sigma[1], sigma[2], sigma[3]
total_travel_time = get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
total_emission = get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
total_revenue = get_total_revenue(tau, sigma_toll, D = D)
print(total_travel_time, total_emission, total_revenue)
print([round(x, 2) for x in sigma])
plt.plot(loss_arr)
plt.title(f"Final Loss = {loss_arr[-1]:.2e}")
plt.show()
