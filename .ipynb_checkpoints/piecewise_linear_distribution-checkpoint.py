import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

## Script Options
DENSITY_RE_CALIBRATE = False
SINGLE_HOUR_RETRAIN = False
TIME_DYNAMIC_RETRAIN = True
N_CPU = 25

## Load Data
df = pd.read_csv("data/df_meta.csv") #pd.read_csv("hourly_demand_20210401.csv")
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df = df.dropna()
df = df[(df["Date"] >= "2021-03-01") & (df["Date"] <= "2021-08-31")]
## Cap speed at 65 mph/hr (i.e. at least 6.61 mins)
# df["Ordinary Travel Time"] = df["Ordinary Travel Time"].apply(lambda x: max(x, 6.61))
# df["HOV Travel Time"] = df["HOV Travel Time"].apply(lambda x: max(x, 6.61))
## Filter out rows where ordinary travel time is not larger than HOV travel time
df = df[df["Ordinary Travel Time"] > df["HOV Travel Time"]]
df_pop["Sigma_2ratio"] = df_pop["TwoPeople"] * 2 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_3ratio"] = df_pop["ThreePlus"] * 3 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df = df.merge(df_pop[["Date", "Sigma_2ratio", "Sigma_3ratio"]], on = "Date")
df = df.sort_values("Date", ascending = True)
TAU_LST = list(df["Avg_total_toll"]) #list(df["Toll"])
HOUR_LST = list(df["Hour"])
LATENCY_O_LST = list(df["Ordinary Travel Time"])
LATENCY_HOV_LST = list(df["HOV Travel Time"])
FLOW_O_LST = list(df["Ordinary Flow"])
FLOW_HOV_LST = list(df["HOV Flow"])
SIGMA_2RATIO_LST = df["Sigma_2ratio"]
SIGMA_3RATIO_LST = df["Sigma_3ratio"]
NUM_DATES = len(df["Date"].unique())
PROFILE_DATE_MAP = torch.zeros((NUM_DATES, df.shape[0]), dtype = torch.float64)
SIGMA_2RATIO_TARGET = torch.zeros(NUM_DATES, dtype = torch.float64)
SIGMA_3RATIO_TARGET = torch.zeros(NUM_DATES, dtype = torch.float64)
date_lst = list(df.drop_duplicates("Date")["Date"])
for i in range(len(date_lst)):
    date = date_lst[i]
    sigma_2ratio = df[df["Date"] == date].iloc[0]["Sigma_2ratio"]
    sigma_3ratio = df[df["Date"] == date].iloc[0]["Sigma_3ratio"]
    idx_lst = np.array(df[df["Date"] == date].index)
    PROFILE_DATE_MAP[i, idx_lst] = 1
    SIGMA_2RATIO_TARGET[i] = sigma_2ratio
    SIGMA_3RATIO_TARGET[i] = sigma_3ratio

## Hyperparameters
VOT_CHUNKS = 2
CARPOOL2_CHUNKS = 4 #10
CARPOOL3_CHUNKS = 2 #10
BETA_RANGE = (0, 1) #0.952
GAMMA2_RANGE = (0, 4) #13.52
GAMMA3_RANGE = (0, 2) #2.71
INT_GRID = 50

## Matches to earlier days closer but shoots up to $14 in the afternoon
# VOT_CHUNKS = 4
# CARPOOL2_CHUNKS = 2 #10
# CARPOOL3_CHUNKS = 4 #10
# BETA_RANGE = (0, 2) #0.952
# GAMMA2_RANGE = (0, 4) #13.52
# GAMMA3_RANGE = (0, 8) #2.71
# INT_GRID = 50

## Latency Parameters
POWER = 4
NUM_LANES = 4
a = 2.4115e-13 #3.2856e-13 #3.9427e-12
b = 0.7906 #0.8789 #10.547
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

def get_total_utility_cost(rho, tau, sigma_o_lst, sigma_toll_lst, sigma_pool2_lst, sigma_pool3_lst, D_lst, beta_lst, gamma2_lst, gamma3_lst, density_lst):
    travel_time_o_lst = get_travel_time_ordinary(rho, np.sum(sigma_o_lst * density_lst), D_lst)
    travel_time_hov_lst = get_travel_time_hov(rho, np.sum(sigma_toll_lst * density_lst), np.sum(sigma_pool2_lst * density_lst), np.sum(sigma_pool3_lst * density_lst), D_lst)
    cost_o = np.sum(sigma_o_lst * density_lst * beta_lst * travel_time_o_lst)
    cost_toll = np.sum(sigma_toll_lst * density_lst * (beta_lst * travel_time_hov_lst + tau))
    cost_pool2 = np.sum(sigma_pool2_lst * density_lst * (beta_lst * travel_time_hov_lst + gamma2_lst + tau / 4))
    cost_pool3 = np.sum(sigma_pool3_lst * density_lst * (beta_lst * travel_time_hov_lst + gamma2_lst + gamma3_lst))
    return cost_o + cost_toll + cost_pool2 + cost_pool3

def get_total_revenue(tau, sigma_toll, sigma_pool2, D):
    return tau * (sigma_toll + 1/4 * sigma_pool2) * D

## Calibrate preference distribution
def get_atomic_strategy_profile(beta, gamma2, gamma3, tau, latency_o, latency_hov, is_scalar = False):
    latency_hov = min(latency_hov, latency_o - 1e-3)
    cost_o = beta * latency_o
    cost_toll = beta * latency_hov + tau
    cost_pool2 = beta * latency_hov + 0.25 * tau + gamma2
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
    c_delta = max(latency_o - latency_hov, 1e-3)
    ## Toll: beta * c_delta > tau, gamma2 > 0.75 tau, gamma2 + gamma3 > tau
    if beta_hi < tau / c_delta or gamma2_hi < 0.75 * tau:
        sigma_toll = 0
    else:
        beta_frac = (beta_hi - max(tau / c_delta, beta_lo)) / (beta_hi - beta_lo)
        ### \int_{max(0.5*tau, \gamma2_lo)}^{\gamma2_hi} \int_{max(\tau-\gamma_2, \gamma3_lo)}^{\gamma3_hi} 1/(gamma2_len*gamma3_len) d\gamma3 d\gamma_2
        gamma2_vec = torch.from_numpy(np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1))
        gamma3_vec = torch.from_numpy(np.linspace(gamma3_lo, gamma3_hi, INT_GRID + 1))
        gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
        gamma3_vec = (gamma3_vec[1:] + gamma3_vec[:-1]) / 2
        gamma23_mat = torch.zeros((INT_GRID, INT_GRID))
        for i in range(INT_GRID):
            gamma2_val = gamma2_vec[i]
            if gamma2_val >= 0.75 * tau:
                lo = torch.max(torch.tensor([tau - gamma2_val, gamma3_lo]))
                gamma23_mat[i,:] += (gamma3_vec >= lo)
        gamma23_frac = torch.mean(gamma23_mat)
        sigma_toll = beta_frac * gamma23_frac
    ## Pool2: beta * c_delta > 0.25 tau + gamma2, gamma2 < 0.75 tau, gamma3 > 0.25 tau
    if gamma2_lo > 0.75 * tau or gamma3_hi < 0.25 * tau:
        sigma_pool2 = 0
    else:
        gamma3_frac = (gamma3_hi - max(0.25 * tau, gamma3_lo)) / (gamma3_hi - gamma3_lo)
        ### \int_{\gamma2_lo}^{min(0.75*tau, \gamma2_hi)} \int_{max(\beta_lo, (0.25*tau+\gamma_2)/c_delta)}^{\beta_hi}
        gamma2_vec = torch.from_numpy(np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1))
        beta_vec = torch.from_numpy(np.linspace(beta_lo, beta_hi, INT_GRID + 1))
        gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
        beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
        gamma2beta_mat = torch.zeros((INT_GRID, INT_GRID))
        for i in range(INT_GRID):
            gamma2_val = gamma2_vec[i]
            if gamma2_val <= 0.75 * tau:
                lo = torch.max(torch.tensor([(0.25*tau+gamma2_val) / c_delta, beta_lo]))
                gamma2beta_mat[i,:] += (beta_vec >= lo)
        gamma2beta_frac = torch.mean(gamma2beta_mat)
        sigma_pool2 = gamma3_frac * gamma2beta_frac
    ## Pool3: beta * c_delta > gamma2 + gamma3, gamma3 < 0.25 tau, gamma2 + gamma3 < tau
    gamma2_vec = torch.from_numpy(np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1))
    gamma3_vec = torch.from_numpy(np.linspace(gamma3_lo, gamma3_hi, INT_GRID + 1))
    beta_vec = torch.from_numpy(np.linspace(beta_lo, beta_hi, INT_GRID + 1))
    gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
    gamma3_vec = (gamma3_vec[1:] + gamma3_vec[:-1]) / 2
    beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
    gamma32beta_mat = torch.zeros((INT_GRID, INT_GRID, INT_GRID))
    for i in range(INT_GRID):
        gamma3_val = gamma3_vec[i]
        if gamma3_val <= 0.25 * tau:
            for j in range(INT_GRID):
                gamma2_val = gamma2_vec[j]
                if gamma2_val + gamma3_val <= tau:
                    lo = (gamma2_val + gamma3_val) / c_delta
                    gamma32beta_mat[i,j,:] += (beta_vec >= lo)
    sigma_pool3 = torch.mean(gamma32beta_mat)
    ## Ordinary lane: beta * c_delta < min(tau, gamma2 + 0.25 tau, gamma2 + gamma3)
    sigma_o = 1 - sigma_toll - sigma_pool2 - sigma_pool3
    # gamma2_vec = torch.from_numpy(np.linspace(gamma2_lo, gamma2_hi, INT_GRID + 1))
    # gamma3_vec = torch.from_numpy(np.linspace(gamma3_lo, gamma3_hi, INT_GRID + 1))
    # beta_vec = torch.from_numpy(np.linspace(beta_lo, beta_hi, INT_GRID + 1))
    # gamma2_vec = (gamma2_vec[1:] + gamma2_vec[:-1]) / 2
    # gamma3_vec = (gamma3_vec[1:] + gamma3_vec[:-1]) / 2
    # beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
    # gamma32beta_mat = torch.zeros((INT_GRID, INT_GRID, INT_GRID))
    # for i in range(INT_GRID):
    #     gamma3_val = gamma3_vec[i]
    #     for j in range(INT_GRID):
    #         gamma2_val = gamma2_vec[j]
    #         lo1 = tau / c_delta
    #         lo2 = (gamma2_val + 0.25 * tau) / c_delta
    #         lo3 = (gamma2_val + gamma3_val) / c_delta
    #         lo = np.min([lo1, lo2, lo3])
    #         gamma32beta_mat[i,j,:] += (beta_vec <= lo)
    # sigma_o = torch.mean(gamma32beta_mat)
    # norm_factor = sigma_o + sigma_toll + sigma_pool2 + sigma_pool3
    # sigma_o = sigma_o / norm_factor
    # sigma_toll = sigma_toll / norm_factor
    # sigma_pool2 = sigma_pool2 / norm_factor
    # sigma_pool3 = sigma_pool3 / norm_factor
    return sigma_o, sigma_toll, sigma_pool2, sigma_pool3

def get_entire_strategy_profile(tau, latency_o, latency_hov):
    beta_vec = np.linspace(BETA_RANGE[0], BETA_RANGE[1], VOT_CHUNKS + 1)
    gamma2_vec = np.linspace(GAMMA2_RANGE[0], GAMMA2_RANGE[1], CARPOOL2_CHUNKS + 1)
    gamma3_vec = np.linspace(GAMMA3_RANGE[0], GAMMA3_RANGE[1], CARPOOL3_CHUNKS + 1)
    sigma_profile = torch.zeros((VOT_CHUNKS * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS, 4))
    for beta_idx in range(VOT_CHUNKS):
        for gamma2_idx in range(CARPOOL2_CHUNKS):
            for gamma3_idx in range(CARPOOL3_CHUNKS):
                beta_lo, beta_hi = beta_vec[beta_idx], beta_vec[beta_idx + 1]
                gamma2_lo, gamma2_hi = gamma2_vec[gamma2_idx], gamma2_vec[gamma2_idx + 1]
                gamma3_lo, gamma3_hi = gamma3_vec[gamma3_idx], gamma3_vec[gamma3_idx + 1]
                sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = get_cube_strategy_profile(beta_lo, beta_hi, gamma2_lo, gamma2_hi, gamma3_lo, gamma3_hi, tau, latency_o, latency_hov)
#                sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = float(sigma_o), float(sigma_toll), float(sigma_pool2), float(sigma_pool3)
                loc = beta_idx * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS + gamma2_idx * CARPOOL3_CHUNKS + gamma3_idx
                sigma_profile[loc, 0] = sigma_o
                sigma_profile[loc, 1] = sigma_toll
                sigma_profile[loc, 2] = sigma_pool2
                sigma_profile[loc, 3] = sigma_pool3
    return sigma_profile

def get_entire_strategy_profile_batch(tau_lst, latency_o_lst, latency_hov_lst):
    data_size = len(tau_lst)
    profile_len = VOT_CHUNKS * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS
    sigma_profile_lst = np.zeros((data_size, profile_len, 4))
    for i in tqdm(range(data_size)):
        sigma_profile_lst[i,:,:] = get_entire_strategy_profile(tau_lst[i], latency_o_lst[i], latency_hov_lst[i]).numpy()
    return sigma_profile_lst

def calibrate_density(tau_lst, latency_o_lst, latency_hov_lst, flow_o_lst, flow_hov_lst, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8):
    assert len(tau_lst) == len(flow_o_lst) and len(tau_lst) == len(flow_hov_lst) and len(tau_lst) == len(latency_o_lst) and len(tau_lst) == len(latency_hov_lst)
    ## Step 1: Construct sigma profiles
    data_size = len(tau_lst)
    profile_len = VOT_CHUNKS * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS
    batch_size = int(math.ceil(data_size / N_CPU))
    ## (i * batch_size):min((i + 1) * batch_size, data_size)
    results = Parallel(n_jobs = N_CPU)(delayed(get_entire_strategy_profile_batch)(
        tau_lst[(i * batch_size):min((i + 1) * batch_size, data_size)], latency_o_lst[(i * batch_size):min((i + 1) * batch_size, data_size)], latency_hov_lst[(i * batch_size):min((i + 1) * batch_size, data_size)]
    ) for i in range(N_CPU))
    sigma_profile_lst_lst = []
    for res in results:
        sigma_profile_lst_lst.append(res)
    sigma_profile_lst = np.concatenate(sigma_profile_lst_lst, axis = 0)
    ## Step 2: Compute density
    tau_lst = torch.tensor(tau_lst)
    flow_o_lst = torch.tensor(flow_o_lst, dtype = torch.float64)
    flow_hov_lst = torch.tensor(flow_hov_lst, dtype = torch.float64)
    latency_o_lst = torch.tensor(latency_o_lst, dtype = torch.float64)
    latency_hov_lst = torch.tensor(latency_hov_lst, dtype = torch.float64)
    sigma_profile_lst = torch.from_numpy(sigma_profile_lst)
    flow_sum_lst = flow_o_lst + flow_hov_lst
    coef_mat = torch.ones((data_size, 4))
    coef_mat[:,0] = -flow_hov_lst / flow_sum_lst
    coef_mat[:,1] = flow_o_lst / flow_sum_lst
    coef_mat[:,2] = 1/2 * flow_o_lst / flow_sum_lst
    coef_mat[:,3] = 1/3 * flow_o_lst / flow_sum_lst
    # hov_sum = PROFILE_DATE_MAP @ flow_hov_lst
    ### Iterate
    f = torch.ones(profile_len, requires_grad = True)
    loss_diff = 100
    loss = 100
    itr = 0
    loss_arr = []
    loss_opt = 100
    f_opt = f.detach().clone()
    while loss > eps and itr < max_itr and eta >= min_eta:
        entire_profiles = sigma_profile_lst.permute(0, 2, 1) * torch.abs(f) #/ torch.sum(torch.abs(f)) #*torch.arange(sigma_profile_lst.ndim - 1, -1, -1)
        entire_profiles_perm = entire_profiles.permute(0, 2, 1)
        entire_profiles_sum = torch.sum(entire_profiles_perm, axis = 1)
        D_lst = (flow_o_lst + flow_hov_lst) / (entire_profiles_sum[:,0] + entire_profiles_sum[:,1] + 1/2 * entire_profiles_sum[:,2] + 1/3 * entire_profiles_sum[:,3])
        # D_lst = (flow_o_lst / entire_profiles_sum[:,0]) + flow_hov_lst / (entire_profiles_sum[:,1] + 1/2 * entire_profiles_sum[:,2] + 1/3 * entire_profiles_sum[:,3]) / 2
        D_sum = PROFILE_DATE_MAP @ D_lst
        # sigma_1 = PROFILE_DATE_MAP @ (D_lst * entire_profiles_sum[:,1])
        flow_2 = PROFILE_DATE_MAP @ (D_lst * entire_profiles_sum[:,2])
        flow_3 = PROFILE_DATE_MAP @ (D_lst * entire_profiles_sum[:,3])
        hov_sum = PROFILE_DATE_MAP @ (D_lst * (entire_profiles_sum[:,1] + entire_profiles_sum[:,2] + entire_profiles_sum[:,3]))
        sigma_2ratio = flow_2 / hov_sum
        sigma_3ratio = flow_3 / hov_sum
        loss_sigma2ratio = torch.mean(torch.abs(sigma_2ratio - SIGMA_2RATIO_TARGET) ** 2)
        loss_sigma3ratio = torch.mean(torch.abs(sigma_3ratio - SIGMA_3RATIO_TARGET) ** 2)
        latency_o = get_travel_time_ordinary(0.25, entire_profiles_sum[:,0], D_lst)
        latency_hov = get_travel_time_hov(0.25, entire_profiles_sum[:,1], entire_profiles_sum[:,2], entire_profiles_sum[:,3], D_lst)
        loss_latency_o = torch.mean(torch.abs(latency_o - latency_o_lst) ** 2)
        loss_latency_hov = torch.mean(torch.abs(latency_hov - latency_hov_lst) ** 2)
        loss = torch.mean(torch.abs(torch.sum(coef_mat * entire_profiles_sum, axis = 1)) ** 2) #* torch.sum(torch.abs(f))
        loss += loss_sigma2ratio + loss_sigma3ratio #+ loss_latency_o + loss_latency_hov
        loss_arr.append(float(loss.data))
        rerun = False
        if loss < loss_opt:
            loss_opt = float(loss.data)
            f_opt = f.detach().clone()
        if len(loss_arr) <= 1:
            loss_diff = 1
        else:
            loss_diff = abs(loss_arr[-1] - loss_arr[-2])
        if loss <= eps:
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
    f = torch.abs(f_opt) / torch.sum(torch.abs(f_opt))
    entire_profiles = sigma_profile_lst.permute(0, 2, 1) * f
    entire_profiles_perm = entire_profiles.permute(0, 2, 1)
    entire_profiles_sum = torch.sum(entire_profiles_perm, axis = 1)
    D_lst = (flow_o_lst + flow_hov_lst) / (entire_profiles_sum[:,0] + entire_profiles_sum[:,1] + 1/2 * entire_profiles_sum[:,2] + 1/3 * entire_profiles_sum[:,3])
#    print(entire_profiles_sum[948,:], D_lst[948])
    print(loss_opt)
    return f.detach().numpy(), D_lst.detach().numpy(), loss_arr, entire_profiles_sum.detach().numpy()

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
def get_equilibrium_profile_atomic(beta_vec, gamma2_vec, gamma3_vec, density_vec, tau, rho, D, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8):
    beta_vec = torch.from_numpy(beta_vec)
    gamma2_vec = torch.from_numpy(gamma2_vec)
    gamma3_vec = torch.from_numpy(gamma3_vec)
    density_vec = torch.from_numpy(density_vec)
    
    if tau == 0:
        sigma_output_vec = np.zeros((len(beta_vec), 4))
        sigma_output_vec[:,0] = 1 - rho
        sigma_output_vec[:,1] = rho
        return np.array([1 - rho, rho, 0, 0]), [0.], sigma_output_vec
    
    sigma_target_init = [0.25, 0.25, 0.25, 0.25]
    sigma_target = torch.tensor(sigma_target_init, requires_grad = True)
    
    loss = 1
    itr = 0
    loss_arr = []
    sigma_opt = sigma_target.clone().detach()
    sigma_opt_vec = None
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
            sigma_opt_vec = sigma_output_vec.T.clone().detach()
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
    if len(loss_arr) == 0:
        loss_arr = [loss_opt]
    return sigma_opt.numpy(), loss_arr, sigma_opt_vec.numpy()

## Optimal toll & capacity design
def get_equilibrium_profile(preference_density, tau, rho, D, max_itr = 100, eta = 0.1, eps = 1e-7, min_eta = 1e-8):
    preference_density = torch.from_numpy(preference_density)
    sigma_target_init = [1 - rho, rho / 3, rho / 3, rho / 3]
    sigma_target = torch.tensor(sigma_target_init, requires_grad = True)
    loss = 1
    itr = 0
    loss_arr = []
    sigma_opt = sigma_target.clone().detach()
    loss_opt = 1
    while loss > eps and itr < max_itr and eta >= min_eta:
        latency_o = get_travel_time_ordinary(rho, sigma_target[0], D)
        latency_hov = get_travel_time_hov(rho, sigma_target[1], sigma_target[2], sigma_target[3], D)
        sigma_output_vec = get_entire_strategy_profile(tau, latency_o, latency_hov)
        sigma_output = torch.sum(sigma_output_vec.T * preference_density, axis = 1)
        loss = torch.sum((sigma_target - sigma_output) ** 2) #+ torch.min(torch.tensor([latency_o - latency_hov, 0])) ** 2
        loss_arr.append(float(loss.data))
        if loss < loss_opt:
            loss_opt = float(loss.data)
            sigma_opt = sigma_target.clone().detach()
        rerun = False
        if loss <= eps:
            break
        if torch.isnan(loss) or itr >= max_itr - 1:# or latency_o < latency_hov:
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

def grid_search_single(rho_vals = [1/4, 2/4, 3/4], toll_lst = [], save_to_file = True, D = None, max_itr = 2000, eta = 1e-3, eps = 1e-3, min_eta = 1e-8):
    total_travel_time_lst = []
    total_emission_lst = []
    total_revenue_lst = []
    total_utility_cost_lst = []
    loss_lst = []
    tau_lst = []
    rho_lst = []
    sigma_o_lst = []
    sigma_toll_lst = []
    sigma_pool2_lst = []
    sigma_pool3_lst = []
    for tau in tqdm(toll_lst):
        for rho in rho_vals:
#            sigma, loss_arr = get_equilibrium_profile(preference_density, tau, rho, D, max_itr = max_itr, eta = eta, eps = eps, min_eta = min_eta)
            sigma, loss_arr, sigma_output_vec = get_equilibrium_profile_atomic(beta_vec, gamma2_vec, gamma3_vec, density_vec, tau, rho, D, max_itr = max_itr, eta = eta, eps = eps, min_eta = min_eta)
            sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = sigma[0], sigma[1], sigma[2], sigma[3]
            total_travel_time = get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
            total_emission = get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
            total_revenue = get_total_revenue(tau, sigma_toll, sigma_pool2, D = D)
            total_utility_cost = get_total_utility_cost(rho, tau, sigma_output_vec[:,0], sigma_output_vec[:,1], sigma_output_vec[:,2], sigma_output_vec[:,3], D, beta_vec, gamma2_vec, gamma3_vec, density_vec)
            
            if len(loss_arr) == 0:
#                print(tau, rho)
                loss_arr = [0.]
            
            total_travel_time_lst.append(total_travel_time)
            total_emission_lst.append(total_emission)
            total_revenue_lst.append(total_revenue)
            total_utility_cost_lst.append(total_utility_cost)
            loss_lst.append(loss_arr[-1])
            tau_lst.append(tau)
            rho_lst.append(rho)
            sigma_o_lst.append(sigma_o)
            sigma_toll_lst.append(sigma_toll)
            sigma_pool2_lst.append(sigma_pool2)
            sigma_pool3_lst.append(sigma_pool3)

    df = pd.DataFrame.from_dict({"% Ordinary": sigma_o_lst, "% Toll": sigma_toll_lst, "% Pool 2": sigma_pool2_lst, "% Pool 3": sigma_pool3_lst, "Total Travel Time": total_travel_time_lst, "Total Emission": total_emission_lst, "Total Revenue": total_revenue_lst, "Total Utility Cost": total_utility_cost_lst, "Loss": loss_lst, "Toll Price": tau_lst, "HOT Capacity": rho_lst})
#    if save_to_file:
#        df.to_csv("opt_3d_results.csv", index = False)
    return df

def grid_search(rho_vals = [1/4, 2/4, 3/4], toll_lst = [], save_to_file = True, D = None, max_itr = 2000, eta = 1e-3, eps = 1e-3, min_eta = 1e-8, n_cpu = 1):
    batch_size = int(math.ceil(len(toll_lst) / n_cpu))
    if n_cpu > 1:
        np.random.shuffle(toll_lst)
        results = Parallel(n_jobs = n_cpu)(delayed(grid_search_single)(
            rho_vals, toll_lst[(i * batch_size):min((i + 1) * batch_size, len(toll_lst))], save_to_file, D, max_itr, eta, eps, min_eta
        ) for i in range(n_cpu))
        df_all = None
        for df in results:
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index = True)
    else:
        df_all = grid_search_single(rho_vals, toll_lst, save_to_file, D, max_itr, eta, eps, min_eta)
    df_all = df_all.sort_values(["Toll Price", "HOT Capacity"], ascending = True)
    if save_to_file:
        df_all.to_csv("opt_3d_results.csv", index = False)
    return df_all

# LATENCY_O_LST = [cost(x / (NUM_LANES - 1)) * DISTANCE for x in FLOW_O_LST]
# LATENCY_HOV_LST = [cost(x / 1) * DISTANCE for x in FLOW_HOV_LST]
# LATENCY_DIFF_LST = [x - y for x,y in zip(LATENCY_O_LST, LATENCY_HOV_LST)]
# print(LATENCY_DIFF_LST)
# print(FLOW_O_LST)
# print(FLOW_HOV_LST)

if DENSITY_RE_CALIBRATE:
    preference_density, D_lst, loss_arr, entire_profiles_sum = calibrate_density(TAU_LST, LATENCY_O_LST, LATENCY_HOV_LST, FLOW_O_LST, FLOW_HOV_LST, max_itr = 5000, eta = 1e-1, eps = 1e-10, min_eta = 1e-5)
    # print([round(x, 3) for x in preference_density])
    # plt.plot(loss_arr)
    # plt.title(f"Final Loss = {loss_arr[-1]:.2e}")
    # plt.savefig("density/preference_loss.png")
    # plt.clf()
    # plt.close()

    with open("density/preference_description.txt", "w") as f:
        beta_vec = np.linspace(BETA_RANGE[0], BETA_RANGE[1], VOT_CHUNKS + 1)
        gamma2_vec = np.linspace(GAMMA2_RANGE[0], GAMMA2_RANGE[1], CARPOOL2_CHUNKS + 1)
        gamma3_vec = np.linspace(GAMMA3_RANGE[0], GAMMA3_RANGE[1], CARPOOL3_CHUNKS + 1)
        for beta_idx in range(VOT_CHUNKS):
            for gamma2_idx in range(CARPOOL2_CHUNKS):
                for gamma3_idx in range(CARPOOL3_CHUNKS):
                    idx = beta_idx * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS + gamma2_idx * CARPOOL3_CHUNKS + gamma3_idx
                    density = preference_density[idx]
                    if density > 1e-3:
                        msg = f"beta[{beta_vec[beta_idx]}, {beta_vec[beta_idx + 1]}], gamma2[{gamma2_vec[gamma2_idx]}, {gamma2_vec[gamma2_idx + 1]}], gamma3[{gamma3_vec[gamma3_idx]}, {gamma3_vec[gamma3_idx + 1]}]: {density}\n"
                        f.write(msg)
    
    df["Demand"] = D_lst
    df["Sigma_o"] = entire_profiles_sum[:,0]
    df["Sigma_toll"] = entire_profiles_sum[:,1]
    df["Sigma_pool2"] = entire_profiles_sum[:,2]
    df["Sigma_pool3"] = entire_profiles_sum[:,3]
    flow_2 = PROFILE_DATE_MAP @ (D_lst * entire_profiles_sum[:,2])
    flow_3 = PROFILE_DATE_MAP @ (D_lst * entire_profiles_sum[:,3])
    hov_sum = PROFILE_DATE_MAP @ (D_lst * (entire_profiles_sum[:,1] + entire_profiles_sum[:,2] + entire_profiles_sum[:,3]))
    sigma_2ratio = flow_2 / hov_sum
    sigma_3ratio = flow_3 / hov_sum
    df_date = df[["Date"]].drop_duplicates()
    df_date["sigma_2ratio_equi"] = sigma_2ratio
    df_date["sigma_3ratio_equi"] = sigma_3ratio
    df_date = df_date.merge(df_pop[["Date", "Sigma_2ratio", "Sigma_3ratio"]], on = "Date")
    df.to_csv("data/df_meta_w_demand.csv", index = False)
    df_date.to_csv("data/df_date_profile.csv", index = False)

    df_hourly_avg = df[["Hour", "HOV Flow", "Ordinary Flow", "HOV Travel Time", "Ordinary Travel Time", "Avg_total_toll", "Demand"]].groupby("Hour").mean().reset_index()
    df_hourly_avg.columns = ["Hour", "HOV Flow", "Ordinary Flow", "HOV Travel Time", "Ordinary Travel Time", "Toll", "Demand"]
    df_hourly_avg.to_csv("data/df_hourly_avg.csv", index = False)

    beta_vec, gamma2_vec, gamma3_vec, density_vec = get_preference_cubes(preference_density)
    preference_density_cubes = np.vstack((beta_vec, gamma2_vec, gamma3_vec, density_vec))
    np.save("density/preference_density.npy", preference_density)
    np.save("density/preference_density_cubes.npy", preference_density_cubes)
else:
    df_hourly_avg = pd.read_csv("data/df_hourly_avg.csv")
    preference_density = np.load("density/preference_density.npy")
    preference_density_cubes = np.load("density/preference_density_cubes.npy")
    preference_len = VOT_CHUNKS * CARPOOL2_CHUNKS * CARPOOL3_CHUNKS
    density_vec = preference_density
#    print(preference_density)
    beta_vec = preference_density_cubes[0,:]
    gamma2_vec = preference_density_cubes[1,:]
    gamma3_vec = preference_density_cubes[2,:]
    density_vec = preference_density_cubes[3,:]

#tau = 0.5
#rho = 0.25
#D = 5801
#
##sigma, loss_arr = get_equilibrium_profile(preference_density, tau, rho, D, max_itr = 2000, eta = 1e-2, eps = 1e-7, min_eta = 1e-8)
#sigma, loss_arr = get_equilibrium_profile_atomic(beta_vec, gamma2_vec, gamma3_vec, density_vec, tau, rho, D, max_itr = 2000, eta = 1e-2, eps = 1e-7, min_eta = 1e-4)
#sigma_o, sigma_toll, sigma_pool2, sigma_pool3 = sigma[0], sigma[1], sigma[2], sigma[3]
#total_travel_time = get_total_travel_time(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
#total_emission = get_total_emission(rho, sigma_o, sigma_toll, sigma_pool2, sigma_pool3, D = D)
#total_revenue = get_total_revenue(tau, sigma_toll, sigma_pool2, D = D)
#print(total_travel_time, total_emission, total_revenue)
#print([round(x, 2) for x in sigma])
#plt.plot(loss_arr)
#plt.title(f"Final Loss = {loss_arr[-1]:.2e}")
#plt.show()

# assert False

GRANULARITY = 0.1

D = df_hourly_avg[df_hourly_avg["Hour"] == 17].iloc[0]["Demand"]
print("Demand =", D)
rho_vals = [1/4, 2/4, 3/4]
if SINGLE_HOUR_RETRAIN:
    df = grid_search(rho_vals = rho_vals, toll_lst = np.arange(0, 15, GRANULARITY), save_to_file = True, D = D, max_itr = 10000, eta = 1e-0, eps = 1e-7, min_eta = 1e-3, n_cpu = N_CPU)
else:
    df = pd.read_csv("opt_3d_results.csv")

### Pipeline for dynamic toll & HOT capacity design
if TIME_DYNAMIC_RETRAIN:
    rho_vals = [1/4] #[1/4, 2/4, 3/4]
    hour_lst = []
    rho_lst = []
    min_congestion_lst = []
    min_congestion_tau_lst = []
    min_emission_lst = []
    min_emission_tau_lst = []
    max_revenue_lst = []
    max_revenue_tau_lst = []
    min_utility_cost_lst = []
    min_utility_cost_tau_lst = []
    df_dynamic_results = None
    for t in tqdm(np.array(df_hourly_avg["Hour"].unique())):
        demand = df_hourly_avg[df_hourly_avg["Hour"] == t].iloc[0]["Demand"]
        df = grid_search(rho_vals = rho_vals, toll_lst = np.arange(0, 15, GRANULARITY), save_to_file = False, D = demand, max_itr = 5000, eta = 1e-1, eps = 1e-7, min_eta = 1e-4, n_cpu = N_CPU)
        if df_dynamic_results is None:
            df_dynamic_results = df
        else:
            df_dynamic_results = pd.concat([df_dynamic_results, df], ignore_index = True)
        for rho in rho_vals:
            df_tmp = df[df["HOT Capacity"] == rho]
            min_congestion = df_tmp["Total Travel Time"].min()
            min_congestion_tau = df_tmp.iloc[df_tmp["Total Travel Time"].argmin()]["Toll Price"]
            min_emission = df_tmp["Total Emission"].min()
            min_emission_tau = df_tmp.iloc[df_tmp["Total Emission"].argmin()]["Toll Price"]
            max_revenue = df_tmp["Total Revenue"].max()
            max_revenue_tau = df_tmp.iloc[df_tmp["Total Revenue"].argmax()]["Toll Price"]
            min_utility_cost = df_tmp["Total Utility Cost"].min()
            min_utility_cost_tau = df_tmp.iloc[df_tmp["Total Utility Cost"].argmin()]["Toll Price"]
            hour_lst.append(t)
            rho_lst.append(rho)
            min_congestion_lst.append(min_congestion)
            min_congestion_tau_lst.append(min_congestion_tau)
            min_emission_lst.append(min_emission)
            min_emission_tau_lst.append(min_emission_tau)
            max_revenue_lst.append(max_revenue)
            max_revenue_tau_lst.append(max_revenue_tau)
            min_utility_cost_lst.append(min_utility_cost)
            min_utility_cost_tau_lst.append(min_utility_cost_tau)

    df_dynamic_results.to_csv("time_dynamic_design_all.csv", index = False)
    df_dynamic = pd.DataFrame.from_dict({"Hour": hour_lst, "Rho": rho_lst, "Min Congestion": min_congestion_lst, "Min Congestion Toll": min_congestion_tau_lst, "Min Emission": min_emission_lst, "Min Emission Toll": min_emission_tau_lst, "Max Revenue": max_revenue_lst, "Max Revenue Toll": max_revenue_tau_lst, "Min Utility Cost": min_utility_cost_lst, "Min Utility Cost Toll": min_utility_cost_tau_lst})
    df_dynamic.to_csv("time_dynamic_design.csv", index = False)
else:
    df_dynamic = pd.read_csv("time_dynamic_design.csv")
