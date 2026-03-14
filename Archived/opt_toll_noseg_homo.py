import json
import math
import itertools
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import scipy
from scipy import optimize
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, csr_array, dia_matrix, vstack
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

## Script Options
N_CPU = 1
DENSITY_RECALIBRATE = False
TRAIN_FRAC = 0.8#0.8
SCHEME = "gamma3=2"

## Hyperparameters
NUM_LANES = 4
BPR_POWER = 4
BPR_A = 7e-4 #2.4115e-13
BPR_B = 0.7906
DISTANCE = 7.16 # miles
WINDOW_SIZE = 1 #15

DELTA = 0.125
num_grids = int(4 / DELTA)

#BETA_RANGE_LST = [(x * DELTA, (x+1) * DELTA) for x in range(num_grids)]
#GAMMA_RANGE_DCT = {
#    1: [(0, 0)],
#    2: [(x * DELTA, (x+1) * DELTA) for x in range(num_grids)],
#    3: [(x * DELTA, (x+1) * DELTA) for x in range(num_grids)]
#}

BETA_RANGE_LST = [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)]
GAMMA_RANGE_DCT = {
    1: [(0, 0)],
    2: [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)],
    3: [(2, 2)]#[(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2)]
}

C = 3
#BETA_RANGE = (BETA_RANGE_LST[0][0], BETA_RANGE_LST[-1][1])
#GAMMA_RANGE_C = [(GAMMA_RANGE_DCT[c][0][0], GAMMA_RANGE_DCT[c][-1][1]) for c in range(1, C + 1)]
INT_GRID = 1 #50

## Load Data
### Date, Hour, Segment, HOV Flow, Ordinary Flow, HOV Travel Time, Ordinary Travel Time, Avg_total_toll
df = pd.read_csv("data/df_meta_5min.csv") #pd.read_csv("hourly_demand_20210401.csv")
# df = df[df["Segment"] == "3460 - Hesperian/238 NB"]
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df = df.sort_values(["Date", "Hour", "Minute"], ascending = True)
#df = df[df["Segment"].isin(['3420 - Auto Mall NB', '3430 - Mowry NB', '3440 - Decoto/84 NB', '3450 - Whipple NB', '3460 - Hesperian/238 NB'])]

data_cols = ['HOV Travel Time', 'Ordinary Travel Time', 'Avg_total_toll'] #['HOV Flow', 'Ordinary Flow', 'HOV Travel Time', 'Ordinary Travel Time', 'Avg_total_toll']
for col in data_cols:
    df[col] = df.groupby(["Segment"])[col].transform(lambda x: x.rolling(WINDOW_SIZE, center = False).mean())
#    df[col] = df.groupby(["Hour", "Segment"])[col].transform(lambda x: x.rolling(WINDOW_SIZE, center = False).mean())
df = df[(df["Date"] >= "2021-02-01") & (df["Date"] <= "2021-05-31")]
df = df[(df["Hour"] >= 14) & (df["Hour"] <= 17)]
df = df.dropna()

df_wide = df.pivot(index = ["Date", "Hour", "Minute"], columns = ["Segment"], values = ["HOV Flow", "Ordinary Flow", "HOV Travel Time", "Ordinary Travel Time", "Avg_total_toll"])
df_wide.columns = [x + "_" + y for x,y in df_wide.columns]
segment_lst = list([x.split("_")[1].strip() for x in df_wide.columns if "HOV Flow" in x])
S = len(segment_lst)
# [14.074  3.165  3.46   2.105  7.16 ]
DISTANCE_ARR = np.zeros(S)
for segment_idx in range(len(segment_lst)):
    distance = df[df["Segment"] == segment_lst[segment_idx]].iloc[0]["Distance"]
    DISTANCE_ARR[segment_idx] = distance
df_wide = df_wide.dropna()
df_wide = df_wide.reset_index()
#df_wide.to_csv("data/df_wide.csv", index = False)

## Cap speed at 65 mph/hr (i.e. at least 6.61 mins)
# df["Ordinary Travel Time"] = df["Ordinary Travel Time"].apply(lambda x: max(x, 6.61))
# df["HOV Travel Time"] = df["HOV Travel Time"].apply(lambda x: max(x, 6.61))
## Filter out rows where ordinary travel time is not larger than HOV travel time
df = df[df["Ordinary Travel Time"] > df["HOV Travel Time"]]
df = df.sort_values(["Date", "Hour", "Minute"], ascending = True)
#data_cols = ['HOV Flow', 'Ordinary Flow', 'HOV Travel Time', 'Ordinary Travel Time', 'Avg_total_toll']
#for col in data_cols:
#    df[col] = df.groupby(["Hour", "Segment"])[col].transform(lambda x: x.rolling(WINDOW_SIZE, center = False).mean())
df_pop["Sigma_1ratio"] = df_pop["Single"] / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_2ratio"] = df_pop["TwoPeople"] * 2 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_3ratio"] = df_pop["ThreePlus"] * 3 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df = df.merge(df_pop[["Date", "Sigma_1ratio", "Sigma_2ratio", "Sigma_3ratio"]], on = "Date")
df = df.sort_values(["Date", "Hour", "Minute"], ascending = True)
df = df.dropna()

TAU_LST = np.array(df["Avg_total_toll"]) #list(df["Toll"])
N_DATA = df_wide.shape[0] #df.shape[0] #100#
TAU_CS_LST = np.zeros((N_DATA, C, S))
LATENCY_O_LST = np.zeros((N_DATA, S))
LATENCY_HOV_LST = np.zeros((N_DATA, S))
FLOW_O_LST = np.zeros(N_DATA * S)
FLOW_HOV_LST = np.zeros(N_DATA * S)
HOUR_LST = np.array(df_wide["Hour"])
N_HOUR = len(df_wide["Hour"].unique())
UNIQUE_HOUR_LST = np.array(df["Hour"].unique())
SEGMENT_LST_ALL = []
HOUR_LST_ALL = []
DATE_LST_ALL = []
### TODO: Change it to multisegments later
for segment_idx in range(len(segment_lst)):
    segment = segment_lst[segment_idx]
    TAU_CS_LST[:,0,segment_idx] = np.array(df_wide[f"Avg_total_toll_{segment}"])
    TAU_CS_LST[:,1,segment_idx] = TAU_CS_LST[:,0,segment_idx] / 4
    LATENCY_O_LST[:,segment_idx] = np.array(df_wide[f"Ordinary Travel Time_{segment}"]) #np.array(df["Ordinary Travel Time"]).reshape((N_DATA, 1))
    LATENCY_HOV_LST[:,segment_idx] = np.array(df_wide[f"HOV Travel Time_{segment}"]) #np.array(df["HOV Travel Time"]).reshape((N_DATA, 1))
    FLOW_O_LST[(N_DATA*segment_idx):(N_DATA*(segment_idx+1))] = np.array(df_wide[f"Ordinary Flow_{segment}"]) #np.array(df["Ordinary Flow"])
    FLOW_HOV_LST[(N_DATA*segment_idx):(N_DATA*(segment_idx+1))] = np.array(df_wide[f"HOV Flow_{segment}"]) #np.array(df["HOV Flow"])
    SEGMENT_LST_ALL += [segment] * N_DATA
    HOUR_LST_ALL += list(df_wide["Hour"])
    DATE_LST_ALL += list(df_wide["Date"])
FLOW_TARGET = np.concatenate((FLOW_O_LST, FLOW_HOV_LST))
LANE_TYPE_ALL = ["Ordinary Lane"] * len(HOUR_LST_ALL) + ["HOT Lane"] * len(HOUR_LST_ALL)
SEGMENT_LST_ALL = SEGMENT_LST_ALL + SEGMENT_LST_ALL
HOUR_LST_ALL = HOUR_LST_ALL + HOUR_LST_ALL
DATE_LST_ALL = DATE_LST_ALL + DATE_LST_ALL
FLOW_COEF = np.ones(len(FLOW_TARGET))
FLOW_COEF[len(FLOW_O_LST):] = 3
segment_type_num = int(S * (S + 1) / 2)
HOUR_OD_DEMAND = np.zeros(N_HOUR * segment_type_num)
df_od_demand = pd.read_csv("data/od_demand.csv")
for hour_idx in range(N_HOUR):
    hour = HOUR_LST[hour_idx]
    segment_idx = 0
    for s_o in range(S):
        origin_seg = segment_lst[s_o]
        for s_d in range(s_o, S):
            dest_seg = segment_lst[s_d]
            HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] = df_od_demand[(df_od_demand["Hour"] == hour) & (df_od_demand["Origin"] == origin_seg) & (df_od_demand["Destination"] == dest_seg)].iloc[0]["Demand"] / 12
            segment_idx += 1
###
#N_DATES = len(df["Date"].unique())
## N_DATES, N_DATA, S
## Days to ignore: 3/31, 4/23, 4/26, 6/30
RATIO_INDEX_TO_IGNORE = [22, 39, 40, 86]
DATES_TO_IGNORE = ["2021-02-15", "2021-03-31", "2021-04-23", "2021-04-26", "2021-04-28", "2021-06-30"]
date_lst = list(set(list(df_wide.drop_duplicates("Date")["Date"])) - set(DATES_TO_IGNORE))
date_lst.sort()
N_DATES = len(date_lst)
PROFILE_DATE_MAP = np.zeros((N_DATES, N_DATA))
RATIO_TARGET = np.zeros((N_DATES, C))
idx = 0
tmp = []
N_DATES_TRAIN = int(N_DATES * TRAIN_FRAC)
N_DATES_TEST = N_DATES - N_DATES_TRAIN
TRAIN_IDX = 0
for i in range(len(date_lst)):
    date = date_lst[i]
    sigma_1ratio = df[df["Date"] == date].iloc[0]["Sigma_1ratio"]
    sigma_2ratio = df[df["Date"] == date].iloc[0]["Sigma_2ratio"]
    sigma_3ratio = df[df["Date"] == date].iloc[0]["Sigma_3ratio"]
    idx_lst = np.array(df_wide[df_wide["Date"] == date].index)
#    if i not in RATIO_INDEX_TO_IGNORE:
#    print(idx, date, PROFILE_DATE_MAP.shape)
    if date not in DATES_TO_IGNORE:
        if idx < N_DATES_TRAIN:
            if len(idx_lst) == 0:
                print(date)
            TRAIN_IDX = max(TRAIN_IDX, max(idx_lst) + 1)
        PROFILE_DATE_MAP[idx, idx_lst] = 1
        RATIO_TARGET[idx, 0] = sigma_1ratio
        RATIO_TARGET[idx, 1] = sigma_2ratio
        RATIO_TARGET[idx, 2] = sigma_3ratio
        idx += 1
        tmp.append(date)
date_lst = tmp
print(date_lst[N_DATES_TRAIN])
TRAIN_TEST = np.zeros(N_DATA)
TRAIN_TEST[:TRAIN_IDX] = 1

class STEArgmin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Get argmin index
        index = torch.argmin(input, dim=-1)
        # Save for backward
        ctx.save_for_backward(input, index)
        return index

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        # Straight-through estimator
        softmin = torch.softmin(input, dim = -1)
        # Apply Jacobian-vector product of softmin:
        # grad_input = J^T @ grad_output, where J is softmin's Jacobian
        dot = (grad_output * softmin).sum(dim=-1, keepdim=True)
        grad_input = softmin * (grad_output - dot)
        return grad_input

def ste_argmin(input):
    return STEArgmin.apply(input)

def get_cost(flow, distance, bpr_a = BPR_A, bpr_b = BPR_B):
    return ((bpr_a * flow) ** BPR_POWER + bpr_b) * distance

def solve_sigma_given_parameters(beta, gamma_c, c_o, c_h, tau_cs):
    C, S = tau_cs.shape
    lane_cs = np.zeros((C, S))
    cost_o = beta * c_o
    cost_h = beta * c_h + gamma_c.reshape((C, 1)) + tau_cs
    lane_cs = (cost_h < c_o) + 0
    total_cost_c = np.sum(lane_cs * cost_h + (1 - lane_cs) * cost_o, axis = 1)
    best_c = np.argmin(total_cost_c)
    return lane_cs[best_c,:]

def solve_sigma_given_parameters_vec_torch(beta_lst, gamma_lst_c, c_o, c_h, tau_cs):
    assert beta_lst.shape[0] == gamma_lst_c.shape[0]
    C, S = tau_cs.shape
    n_grids = beta_lst.shape[0]
    beta_lst = beta_lst.reshape((1, n_grids, 1, 1))
    segment_type_num = int(S * (S + 1) / 2)
    gamma_lst_c = gamma_lst_c.reshape((1, n_grids, C, 1))
    n_data = 1#len(c_o)
    c_o = c_o.reshape((n_data, 1, 1, S))
    c_h = c_h.reshape((n_data, 1, 1, S))
    tau_cs = tau_cs.reshape((n_data, 1, C, S))
    cost_o = beta_lst * c_o
    cost_h = beta_lst * c_h + gamma_lst_c + tau_cs
    lane_cs = (cost_h < cost_o) + 0
    total_cost_mat = lane_cs * cost_h + (1 - lane_cs) * cost_o #np.sum(lane_cs * cost_h + (1 - lane_cs) * cost_o, axis = 3)
    total_cost_c_lst = []
    best_c_lst = []
    for s_o in range(S):
        for s_d in range(s_o, S):
            total_cost_c = total_cost_mat[:,:,:, s_o:(s_d+1)].sum(dim = 3)
            best_c = ste_argmin(total_cost_c)
            total_cost_c_lst.append(total_cost_c)
            best_c_lst.append(best_c)
#    best_c = np.argmin(total_cost_c, axis = 2)
    lane_cs_h = torch.zeros((n_data, n_grids, segment_type_num, C, S))
    lane_cs_o = torch.zeros((n_data, n_grids, segment_type_num, C, S))
    for data_idx in tqdm(range(n_data), leave = False):
        for grid_idx in tqdm(range(n_grids), leave = False):
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    best_c = best_c_lst[segment_idx][data_idx, grid_idx]
                    lane_cs_h[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    lane_cs_o[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = 1 - lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    segment_idx += 1
    return lane_cs_h, lane_cs_o #lane_cs[:,best_c,:]

def solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, c_o, c_h, tau_cs):
    assert beta_lst.shape[0] == gamma_lst_c.shape[0]
    C, S = tau_cs.shape
    n_grids = beta_lst.shape[0]
    beta_lst = beta_lst.reshape((1, n_grids, 1, 1))
    segment_type_num = int(S * (S + 1) / 2)
    gamma_lst_c = gamma_lst_c.reshape((1, n_grids, C, 1))
    n_data = 1#len(c_o)
    c_o = c_o.reshape((n_data, 1, 1, S))
    c_h = c_h.reshape((n_data, 1, 1, S))
    tau_cs = tau_cs.reshape((n_data, 1, C, S))
    cost_o = beta_lst * c_o
    cost_h = beta_lst * c_h + gamma_lst_c + tau_cs
    lane_cs = (cost_h < cost_o) + 0
    total_cost_mat = lane_cs * cost_h + (1 - lane_cs) * cost_o #np.sum(lane_cs * cost_h + (1 - lane_cs) * cost_o, axis = 3)
    total_cost_c_lst = []
    best_c_lst = []
    for s_o in range(S):
        for s_d in range(s_o, S):
            total_cost_c = total_cost_mat[:,:,:, s_o:(s_d+1)].sum(axis = 3)
            best_c = np.argmin(total_cost_c, axis = 2)
            total_cost_c_lst.append(total_cost_c)
            best_c_lst.append(best_c)
#    best_c = np.argmin(total_cost_c, axis = 2)
    lane_cs_h = np.zeros((n_data, n_grids, segment_type_num, C, S))
    lane_cs_o = np.zeros((n_data, n_grids, segment_type_num, C, S))
    for data_idx in tqdm(range(n_data), leave = False):
        for grid_idx in tqdm(range(n_grids), leave = False):
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    best_c = best_c_lst[segment_idx][data_idx, grid_idx]
                    lane_cs_h[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    lane_cs_o[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = 1 - lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    segment_idx += 1
    return lane_cs_h, lane_cs_o #lane_cs[:,best_c,:]

def elem_in_range(beta, gamma_c, lst):
    eps = 1e-9
    if beta > lst[0][1] + eps:
        return False
    for c in range(C):
        if gamma_c[c] > lst[c + 1][1] + eps:
            return False
    return True

def get_beta_gamma_range_lst(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT):
    beta_gamma_range_lst = [[x] for x in beta_range_lst]
    for c in range(C):
        tmp = []
        for lst in beta_gamma_range_lst:
            for tup in gamma_range_dct[c + 1]:
                elem = lst.copy() + [tup]
                tmp.append(elem)
        beta_gamma_range_lst = tmp
    return beta_gamma_range_lst

def get_d_idx_map_v2(beta_lst, gamma_lst_c, beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT):
    assert len(beta_lst) == gamma_lst_c.shape[0]
    beta_gamma_range_lst = get_beta_gamma_range_lst(beta_range_lst = beta_range_lst, gamma_range_dct = gamma_range_dct)
    d_num = len(beta_gamma_range_lst)
    d_idx_start_lst = np.zeros(d_num + 1)
    idx = 0
    for i in range(len(beta_lst)):
        beta = beta_lst[i]
        gamma_c = gamma_lst_c[i,:]
        lst = beta_gamma_range_lst[idx]
        if not elem_in_range(beta, gamma_c, lst):
            idx += 1
            d_idx_start_lst[idx] = i
    d_idx_start_lst[-1] = len(beta_lst)
    return d_idx_start_lst.astype(int)

def get_grid(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT):
#    beta_vec = np.linspace(BETA_RANGE[0], BETA_RANGE[1], INT_GRID + 1)
#    gamma_mat = np.zeros((C, INT_GRID + 1))
#    for c in range(1, C):
#        gamma_c_grid = np.linspace(GAMMA_RANGE_C[c][0], GAMMA_RANGE_C[c][1], INT_GRID + 1)
#        gamma_mat[c,:] = gamma_c_grid
#    beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
#    gamma_mat = (gamma_mat[:,1:] + gamma_mat[:,:-1]) / 2
    beta_gamma_range_lst = get_beta_gamma_range_lst(beta_range_lst = beta_range_lst, gamma_range_dct = gamma_range_dct)
    beta_lst = []
    gamma_lst_c = []
    eps = 1e-9
    for lst in beta_gamma_range_lst:
        beta_curr = np.arange(lst[0][0], lst[0][1] + eps, DELTA) #np.linspace(lst[0][0], lst[0][1], INT_GRID + 1) #beta_vec[(beta_vec > lst[0][0]) & (beta_vec <= lst[0][1])]
        gamma_c_curr = []
        for c in range(1, C):
            tmp = np.arange(lst[c+1][0], lst[c+1][1] + eps, DELTA) #np.linspace(lst[c+1][0], lst[c+1][1], INT_GRID + 1) #gamma_mat[c,:][(gamma_mat[c,:] > lst[c+1][0]) & (gamma_mat[c,:] <= lst[c+1][1])]
            gamma_c_curr.append(tmp)
        grid_tup = [x.ravel() for x in np.meshgrid(beta_curr, *gamma_c_curr, indexing = "ij")]
        beta_lst_curr = grid_tup[0]
        gamma_lst_c_curr = np.vstack(grid_tup[1:]).T
        gamma_lst_c_curr = np.hstack((np.zeros((gamma_lst_c_curr.shape[0], 1)), gamma_lst_c_curr))
        beta_lst.append(beta_lst_curr)
        gamma_lst_c.append(gamma_lst_c_curr)
    beta_lst = np.concatenate(beta_lst)
    gamma_lst_c = np.concatenate(gamma_lst_c)
    d_idx_start_lst = get_d_idx_map_v2(beta_lst, gamma_lst_c, beta_range_lst = beta_range_lst, gamma_range_dct = gamma_range_dct)
    gamma_lst_c = gamma_lst_c.cumsum(axis = 1)
    return beta_lst, gamma_lst_c, d_idx_start_lst

def profile_given_data_single(lo, hi, beta_lst, gamma_lst_c, segment_type_num, latency_o_lst = LATENCY_O_LST, latency_hov_lst = LATENCY_HOV_LST, tau_cs_lst = TAU_CS_LST):
    N_DATA, C, S = tau_cs_lst.shape
    sigma_ns_h = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S), dtype=np.uint8)
    sigma_ns_o = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S), dtype=np.uint8)
    for data_idx in tqdm(range(lo, hi)):
        sigma_s_h, sigma_s_o = solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, latency_o_lst[data_idx,:], latency_hov_lst[data_idx,:], tau_cs_lst[data_idx,:,:])
        sigma_s_h, sigma_s_o = sigma_s_h.astype(np.uint8), sigma_s_o.astype(np.uint8)
        sigma_ns_h[data_idx,:,:,:,:] = sigma_s_h[0,:,:,:,:]
        sigma_ns_o[data_idx,:,:,:,:] = sigma_s_o[0,:,:,:,:]
    return sigma_ns_h, sigma_ns_o

def get_d_coef_matrix(sigma_ns_h, sigma_ns_o, meta_data = None, data_dct = None):
    global N_HOUR, S, C, BETA_RANGE_LST, GAMMA_RANGE_DCT, HOUR_OD_DEMAND, N_DATA, HOUR_LST_ALL, HOUR_LST, UNIQUE_HOUR_LST
    if meta_data is not None:
        N_HOUR = meta_data["N_HOUR"]
        S = meta_data["S"]
        C = meta_data["C"]
        BETA_RANGE_LST = meta_data["BETA_RANGE_LST"]
        GAMMA_RANGE_DCT = meta_data["GAMMA_RANGE_DCT"]
        HOUR_OD_DEMAND = meta_data["HOUR_OD_DEMAND"]
    if data_dct is not None:
        N_DATA = data_dct["N_DATA"]
        HOUR_LST_ALL = data_dct["HOUR_LST_ALL"]
        HOUR_LST = data_dct["HOUR_LST"]
        UNIQUE_HOUR_LST = data_dct["UNIQUE_HOUR_LST"]
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT)
    segment_type_num = int(S * (S + 1) / 2)
    ## Compute equilibrium flow using d
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len)
    ### Compute equilibrium flows
    ## TODO: Implement d_to_f_mat
    ### o + h
    d_coef_matrix = np.zeros((2 * N_DATA + 1, d_len))
    for hour_idx in tqdm(range(N_HOUR)):
        t = UNIQUE_HOUR_LST[hour_idx]
        relev_data_idx = np.where(HOUR_LST == t)[0]
        for d_idx in range(single_t_d_len):
            elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    for s in range(s_o, s_d + 1):
                        for c in range(C):
                            d_coef_matrix[relev_data_idx, hour_idx * single_t_d_len + d_idx] += 1 / (c + 1) * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            d_coef_matrix[N_DATA + relev_data_idx, hour_idx * single_t_d_len + d_idx] += 1 / (c + 1) * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                    segment_idx += 1
    d_coef_matrix[-1,:] = 1
    return d_coef_matrix

def drop_dependent_columns(X, tol=1e-10):
    """
    Drop linearly dependent columns from matrix X.
    
    Parameters:
        X (np.ndarray): An (n x p) matrix.
        tol (float): Tolerance threshold for determining linear dependence.
        
    Returns:
        X_indep (np.ndarray): Matrix with linearly independent columns.
        idx_indep (list): Indices of independent columns kept.
    """
    # Perform QR decomposition with column pivoting
    Q, R, pivot = scipy.linalg.qr(X, mode='economic', pivoting=True)
    
    # Determine rank based on tolerance
    diag_R = np.abs(np.diag(R))
    rank = np.sum(diag_R > tol)
    
    # Select only the independent columns
    idx_indep = sorted(pivot[:rank])
    idx_dropped = sorted(pivot[rank:])
    X_indep = X[:, idx_indep]
    
    return X_indep, idx_dropped

def is_identifiable(sigma_ns_h, sigma_ns_o, meta_data = None, data_dct = None):
    d_coef_matrix = get_d_coef_matrix(sigma_ns_h, sigma_ns_o, meta_data = meta_data, data_dct = data_dct)
    mat_rank = np.linalg.matrix_rank(d_coef_matrix)
    print(mat_rank, d_coef_matrix.shape)
    d_coef_matrix_shorter, d_idx_dropped = drop_dependent_columns(d_coef_matrix)
#    mat_rank = np.linalg.matrix_rank(d_coef_matrix_shorter)
#    print(mat_rank, d_coef_matrix_shorter.shape)
#    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
#    segment_type_num = int(S * (S + 1) / 2)
#    ## Compute equilibrium flow using d
#    single_t_d_len = len(d_idx_start_lst) - 1
#    d_coef_idx = 0
#    for t in range(N_HOUR):
#        for d_idx in range(single_t_d_len):
#            for segment_idx in range(segment_type_num):
#                if d_coef_idx in d_idx_dropped:
#                    print(t, d_idx, segment_idx)
#                d_coef_idx += 1
#    assert False
#    return d_idx_dropped
    return d_idx_dropped

def optimize_density(d_len, d_to_f_mat, d_to_fh_mat, d_to_fh_total_mat, single_t_d_len, d_idx_dropped):
    model = gp.Model()
    d = model.addMVar(d_len, lb = 0, vtype = GRB.CONTINUOUS, name = "d")
    ### Compute equilibrium flows
    f_equi = model.addMVar(2 * N_DATA * S, lb = 0, vtype = GRB.CONTINUOUS, name = "f")
    f_h_equi = model.addMVar(C * N_DATA, lb = 0, vtype = GRB.CONTINUOUS, name = "fh")
    f_h_total_equi = model.addMVar(N_DATA, lb = 0, vtype = GRB.CONTINUOUS, name = "fh_total")
    model.addConstr(d_to_f_mat @ d == f_equi)
    model.addConstr(d_to_fh_mat @ d == f_h_equi)
    model.addConstr(d_to_fh_total_mat @ d == f_h_total_equi)
#    for d_idx in d_idx_dropped:
#        model.addConstr(d[d_idx] == 0)
    for hour_idx in range(N_HOUR):
        density_expr = gp.LinExpr(0.0)
        for k in range(single_t_d_len):
            d_col = hour_idx * single_t_d_len + k
            density_expr += d[d_col]
        model.addConstr(density_expr == 1)
    ### Compute objective function
    ## Ordinary lanes:
    obj_ordinary = ((f_equi[:(TRAIN_IDX * S)] - FLOW_TARGET[:(TRAIN_IDX * S)]) * (f_equi[:(TRAIN_IDX * S)] - FLOW_TARGET[:(TRAIN_IDX * S)])).sum() / TRAIN_IDX
    obj_hot = ((f_equi[(N_DATA * S):(N_DATA * S + TRAIN_IDX * S)] - FLOW_TARGET[(N_DATA * S):(N_DATA * S + TRAIN_IDX * S)]) * (f_equi[(N_DATA * S):(N_DATA * S + TRAIN_IDX * S)] - FLOW_TARGET[(N_DATA * S):(N_DATA * S + TRAIN_IDX * S)])).sum() / TRAIN_IDX
    objective = obj_ordinary + obj_hot * 9
#    objective = ((f_equi[:(2 * TRAIN_IDX * S)] - FLOW_TARGET[:(2 * TRAIN_IDX * S)]) * FLOW_COEF[:(2 * TRAIN_IDX * S)] * (f_equi[:(2 * TRAIN_IDX * S)] - FLOW_TARGET[:(2 * TRAIN_IDX * S)]) * FLOW_COEF[:(2 * TRAIN_IDX * S)]).sum() / TRAIN_IDX
#    objective = ((f_equi - FLOW_TARGET) * FLOW_COEF * (f_equi - FLOW_TARGET) * FLOW_COEF).sum() / N_DATA
    ### Compute ratios of each toll class
    ratio_idx = [i for i in range(len(date_lst)) if i not in RATIO_INDEX_TO_IGNORE]
    flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_equi
    ### Add constraints on lower bound of daily flow to avoid trivial solutions
    all_seg_flow = 0
    for s in range(S):
        all_seg_flow += FLOW_TARGET[(N_DATA * S + s)::S]
    daily_flow_lb = PROFILE_DATE_MAP @ all_seg_flow
    ratio_total = 0
    for c in range(C):
        ratio_total += 1 / (c + 1) * RATIO_TARGET[:,c] * flow_ratio_target_total
        ratio_loss = (PROFILE_DATE_MAP[:N_DATES_TRAIN,:TRAIN_IDX] @ f_h_equi[(c*N_DATA):(c*N_DATA + TRAIN_IDX)] - RATIO_TARGET[:N_DATES_TRAIN,c] * flow_ratio_target_total[:N_DATES_TRAIN]) #/ N_HOUR
        objective += (ratio_loss * ratio_loss).sum() / TRAIN_IDX * 10
#        ratio_loss = (PROFILE_DATE_MAP @ f_h_equi[(c*N_DATA):((c+1)*N_DATA)] - RATIO_TARGET[:,c] * flow_ratio_target_total) #/ N_HOUR
#        objective += (ratio_loss * ratio_loss).sum() / N_DATA * 10
    ### Optimize the model
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    obj_val = model.ObjVal
    f_h_ret = np.zeros(C * N_DATA)
    for i in range(C * N_DATA):
        f_h_ret[i] = f_h_equi[i].x
    f_h_total_ret = np.zeros(N_DATA)
    for i in range(N_DATA):
        f_h_total_ret[i] = f_h_total_equi[i].x
    density = np.zeros(d_len)
    for i in range(d_len):
        density[i] = d[i].x
    return obj_val, density, f_h_ret, f_h_total_ret

def calibrate_density():
    ## Get sigma profile for each grid
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    segment_type_num = int(S * (S + 1) / 2)
    ### Compute profile given data
    if N_CPU > 1:
        sigma_ns_h = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
        sigma_ns_o = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
        batch_size = int(math.ceil(N_DATA / N_CPU))
        results = Parallel(n_jobs = N_CPU)(delayed(profile_given_data_single)(
            i * batch_size, min(N_DATA, (i + 1) * batch_size), beta_lst, gamma_lst_c, segment_type_num
        ) for i in range(N_CPU))
        for res in tqdm(results):
            sigma_ns_h += res[0]
            sigma_ns_o += res[1]
    else:
        sigma_ns_h, sigma_ns_o = profile_given_data_single(0, N_DATA, beta_lst, gamma_lst_c, segment_type_num)
    d_idx_dropped = is_identifiable(sigma_ns_h, sigma_ns_o)
    ## Compute equilibrium flow using d
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len)
    ## TODO: Implement d_to_f_mat
    ### o + h
    d_to_f_mat = np.zeros((2 * N_DATA * S, d_len))
    d_to_fh_mat = np.zeros((C * N_DATA, d_len))
    d_to_fh_total_mat = np.zeros((N_DATA, d_len))
    for hour_idx in tqdm(range(N_HOUR)):
        t = UNIQUE_HOUR_LST[hour_idx]
        relev_data_idx = np.where(HOUR_LST == t)[0]
        for d_idx in range(single_t_d_len):
            elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    for s in range(s_o, s_d + 1):
                        for c in range(C):
                            d_to_f_mat[relev_data_idx * S + s, hour_idx * single_t_d_len + d_idx] += 1 / (c + 1) * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            d_to_f_mat[N_DATA * S + relev_data_idx * S + s, hour_idx * single_t_d_len + d_idx] += 1 / (c + 1) * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] #/ C #/ (s_d - s_o + 1)
                            d_to_fh_mat[c * N_DATA + relev_data_idx, hour_idx * single_t_d_len + d_idx] += sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] #/ C #/ (s_d - s_o + 1)
                            d_to_fh_total_mat[relev_data_idx, hour_idx * single_t_d_len + d_idx] += sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] #/ C #/ (s_d - s_o + 1)
                    segment_idx += 1
    obj_val, density, f_h_ret, f_h_total_ret = optimize_density(d_len, d_to_f_mat, d_to_fh_mat, d_to_fh_total_mat, single_t_d_len, d_idx_dropped)
    f_equi_ret = d_to_f_mat @ density
    df_tmp = pd.DataFrame.from_dict({"Flow Equi": f_equi_ret, "Flow Target": FLOW_TARGET})
    df_tmp["Lane Type"] = LANE_TYPE_ALL
    df_tmp["Date"] = DATE_LST_ALL
    df_tmp["Hour"] = HOUR_LST_ALL
    df_tmp["Segment"] = SEGMENT_LST_ALL
    df_tmp.to_csv(f"CalibrationResults/Flow/tmp_{SCHEME}.csv", index = False)
    dct_ratio = {"Date": date_lst}
    flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_ret
    for c in range(C):
        dct_ratio[f"Equi {c}"] = PROFILE_DATE_MAP @ f_h_ret[(c*N_DATA):((c+1)*N_DATA)]
        dct_ratio[f"Target {c}"] = RATIO_TARGET[:,c] * flow_ratio_target_total
    df_tmp_ratio = pd.DataFrame.from_dict(dct_ratio)
    df_tmp_ratio.to_csv(f"CalibrationResults/Flow/tmp_ratio_{SCHEME}.csv", index = False)
    return density

def describe_density(density, meta_data = None):
    global N_HOUR, S, C, BETA_RANGE_LST, GAMMA_RANGE_DCT, HOUR_OD_DEMAND, UNIQUE_HOUR_LST
    if meta_data is not None:
        N_HOUR = meta_data["N_HOUR"]
        S = meta_data["S"]
        C = meta_data["C"]
        BETA_RANGE_LST = meta_data["BETA_RANGE_LST"]
        GAMMA_RANGE_DCT = meta_data["GAMMA_RANGE_DCT"]
        HOUR_OD_DEMAND = meta_data["HOUR_OD_DEMAND"]
        UNIQUE_HOUR_LST = meta_data["UNIQUE_HOUR_LST"]
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT)
    beta_gamma_range_lst = get_beta_gamma_range_lst(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT)
    segment_type_num = int(S * (S + 1) / 2)
    segment_range_lst = []
    segment_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            name = f"{segment_lst[s_o]} to {segment_lst[s_d]}"
            segment_range_lst.append(name)
            segment_idx += 1
    single_t_d_len = len(d_idx_start_lst) - 1
    for hour_idx in range(N_HOUR):
        t = UNIQUE_HOUR_LST[hour_idx]
        print(f"Hour = {t}:")
        for segment_idx in range(segment_type_num):
            print(f"\tSegment type = {segment_range_lst[segment_idx]}:")
            for d_idx in range(single_t_d_len):
                val = density[hour_idx * single_t_d_len + d_idx] * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                tup = beta_gamma_range_lst[d_idx]
                if val > 1e-3:
                    print(f"\t\tBeta = {tup[0]}, Gamma = {tup[1:]}: {val}")

def get_segment_pop(density, hour_idx):
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    segment_pop = np.zeros(segment_type_num)
    for segment_type_idx in range(segment_type_num):
        density_idx_begin = hour_idx * single_t_d_len * segment_type_num + segment_type_idx
        density_idx_end = (hour_idx + 1) * single_t_d_len * segment_type_num
        pop = density[density_idx_begin:density_idx_end:segment_type_num].sum()
        segment_pop[segment_type_idx] = pop
    return segment_pop


density = calibrate_density()
#describe_density(density)
