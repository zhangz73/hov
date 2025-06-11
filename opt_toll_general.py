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
from joblib import Parallel, delayed
from tqdm import tqdm

## Script Options
N_CPU = 1
DENSITY_RECALIBRATE = True
TRAIN_FRAC = 0.8#0.8

## Hyperparameters
NUM_LANES = 4
BPR_POWER = 4
BPR_A = 7e-4 #2.4115e-13
BPR_B = 0.7906
WINDOW_SIZE = 5 #15

BETA_RANGE_LST = [(0, 0.1), (1, 2)]
GAMMA_RANGE_DCT = {
    1: [(0, 0)],
    2: [(0, 0.1), (3, 4)],
    3: [(0, 0.1), (3, 4)]
}
#BETA_GAMMA_RANGE_LST = [
#    [(0, 0.1), (0, 0), (3, 4), (3, 4)],
#    [(1, 2), (0, 0), (0, 0.1), (0, 0.1)],
#    [(1, 2), (0, 0), (0, 0.1), (3, 4)],
#    [(1, 2), (0, 0), (3, 4), (3, 4)]
#]
#BETA_RANGE_LST = [(0, 1), (1, 2)]
#GAMMA_RANGE_DCT = {
#    1: [(0, 0)],
#    2: [(0, 0.25), (0.25, 2), (2, 4)],
#    3: [(0, 0.25), (0.25, 1), (1, 2)]
#}
C = 3
BETA_RANGE = (BETA_RANGE_LST[0][0], BETA_RANGE_LST[-1][1])
GAMMA_RANGE_C = [(GAMMA_RANGE_DCT[c][0][0], GAMMA_RANGE_DCT[c][-1][1]) for c in range(1, C + 1)]
INT_GRID = 5 #50

## Load Data
### Date, Hour, Segment, HOV Flow, Ordinary Flow, HOV Travel Time, Ordinary Travel Time, Avg_total_toll
df = pd.read_csv("data/df_meta.csv") #pd.read_csv("hourly_demand_20210401.csv")
# df = df[df["Segment"] == "3460 - Hesperian/238 NB"]
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df = df.dropna()
df = df[(df["Date"] >= "2021-03-01") & (df["Date"] <= "2021-08-31")]
#df = df[(df["Hour"] >= 12) & (df["Hour"] <= 18)]
#df = df[df["Segment"].isin(['3420 - Auto Mall NB', '3430 - Mowry NB', '3440 - Decoto/84 NB', '3450 - Whipple NB', '3460 - Hesperian/238 NB'])]
#df_new = df.copy()
#for col in ["HOV Travel Time", "Ordinary Travel Time", "Avg_total_toll"]:
#    df_new[col] += np.random.normal(0, 1, size = df_new.shape[0]) * 0.1
#df_new["Date"] = (pd.to_datetime(df["Date"]) + pd.DateOffset(months=6)).dt.strftime("%Y-%m-%d")
#df = pd.concat([df, df_new], axis = 0, ignore_index = True).reset_index()
#df = df.drop_duplicates(subset = ["Date", "Hour", "Segment"])

## Detrend the demand
#df["demand"] = 0
#for col in df.columns:
#    if "Flow" in col:
#        df["demand"] += df[col]
#df_demand = df[["Date", "demand"]].groupby(["Date"]).sum().reset_index().sort_values("Date")
#slope_intercept = np.polyfit(np.arange(df_demand.shape[0]), df_demand["demand"], 1)
#demand_slope, demand_intercept = slope_intercept[0], slope_intercept[1]
#df_demand["detrend_coef"] = demand_intercept / (demand_intercept + demand_slope * np.arange(df_demand.shape[0]))
#df = df.merge(df_demand[["Date", "detrend_coef"]], on = "Date")
#for col in df.columns:
#    if "Flow" in col:
#        df[col] = df[col] * df["detrend_coef"]
#df_flow = df[["Date"] + [x for x in df.columns if "Flow" in x]].copy()
#df_flow = df_flow.groupby(["Date"]).sum().reset_index()
#df_detrend_coef = df_flow[["Date"]].copy()
#for col in df_flow:
#    if "Flow" in col:
#        slope_intercept = np.polyfit(np.arange(df_flow.shape[0]), df_flow[col], 1)
#        detrend_coef = slope_intercept[1] / (slope_intercept[1] + slope_intercept[0] * np.arange(df_flow.shape[0]))
#        print(detrend_coef)
##        df_flow[col] = df_flow[col] * detrend_coef
##        df_flow[f"detrend_coef_{col}"] = detrend_coef
#        df_detrend_coef[f"detrend_coef_{col}"] = detrend_coef
#df = df.merge(df_detrend_coef, on = ["Date"])
#for col in df.columns:
#    if "Flow" in col and "detrend" not in col:
#        df[col] = df[col] * df[f"detrend_coef_{col}"]

data_cols = ['HOV Flow', 'Ordinary Flow', 'HOV Travel Time', 'Ordinary Travel Time', 'Avg_total_toll']
for col in data_cols:
    df[col] = df.groupby(["Hour", "Segment"])[col].transform(lambda x: x.rolling(WINDOW_SIZE, center = False).mean())

df_wide = df.pivot(index = ["Date", "Hour"], columns = ["Segment"], values = ["HOV Flow", "Ordinary Flow", "HOV Travel Time", "Ordinary Travel Time", "Avg_total_toll"])
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
df = df.sort_values(["Date", "Hour"], ascending = True)
#data_cols = ['HOV Flow', 'Ordinary Flow', 'HOV Travel Time', 'Ordinary Travel Time', 'Avg_total_toll']
#for col in data_cols:
#    df[col] = df.groupby(["Hour", "Segment"])[col].transform(lambda x: x.rolling(WINDOW_SIZE, center = False).mean())
df_pop["Sigma_1ratio"] = df_pop["Single"] / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_2ratio"] = df_pop["TwoPeople"] * 2 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_3ratio"] = df_pop["ThreePlus"] * 3 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df = df.merge(df_pop[["Date", "Sigma_1ratio", "Sigma_2ratio", "Sigma_3ratio"]], on = "Date")
df = df.sort_values(["Date", "Hour"], ascending = True)
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
FLOW_TARGET = np.concatenate((FLOW_O_LST, FLOW_HOV_LST))
LANE_TYPE_ALL = ["Ordinary Lane"] * len(HOUR_LST_ALL) + ["HOT Lane"] * len(HOUR_LST_ALL)
SEGMENT_LST_ALL = SEGMENT_LST_ALL + SEGMENT_LST_ALL
HOUR_LST_ALL = HOUR_LST_ALL + HOUR_LST_ALL
FLOW_COEF = np.ones(len(FLOW_TARGET))
FLOW_COEF[len(FLOW_O_LST):] = 3
###
#N_DATES = len(df["Date"].unique())
## N_DATES, N_DATA, S
## Days to ignore: 3/31, 4/23, 4/26, 6/30
RATIO_INDEX_TO_IGNORE = [22, 39, 40, 86]
DATES_TO_IGNORE = ["2021-03-31", "2021-04-23", "2021-04-26", "2021-06-30"]
date_lst = list(set(list(df.drop_duplicates("Date")["Date"])) - set(DATES_TO_IGNORE))
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
            TRAIN_IDX = max(TRAIN_IDX, max(idx_lst) + 1)
        PROFILE_DATE_MAP[idx, idx_lst] = 1
        RATIO_TARGET[idx, 0] = sigma_1ratio
        RATIO_TARGET[idx, 1] = sigma_2ratio
        RATIO_TARGET[idx, 2] = sigma_3ratio
        idx += 1
        tmp.append(date)
date_lst = tmp
print(date_lst[N_DATES_TRAIN])

def get_cost(flow, distance):
    return ((BPR_A * flow) ** BPR_POWER + BPR_B) * distance

def solve_sigma_given_parameters(beta, gamma_c, c_o, c_h, tau_cs):
    lane_cs = np.zeros((C, S))
    cost_o = beta * c_o
    cost_h = beta * c_h + gamma_c.reshape((C, 1)) + tau_cs
    lane_cs = (cost_h < c_o) + 0
    total_cost_c = np.sum(lane_cs * cost_h + (1 - lane_cs) * cost_o, axis = 1)
    best_c = np.argmin(total_cost_c)
    return lane_cs[best_c,:]

def solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, c_o, c_h, tau_cs):
    assert beta_lst.shape[0] == gamma_lst_c.shape[0]
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
    if beta > lst[0][1]:
        return False
    for c in range(C):
        if gamma_c[c] > lst[c + 1][1]:
            return False
    return True

def get_beta_gamma_range_lst(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT):
#    return BETA_GAMMA_RANGE_LST
    beta_gamma_range_lst = [[x] for x in beta_range_lst]
    for c in range(C):
        tmp = []
        for lst in beta_gamma_range_lst:
            for tup in gamma_range_dct[c + 1]:
                elem = lst.copy() + [tup]
                tmp.append(elem)
        beta_gamma_range_lst = tmp
    return beta_gamma_range_lst

def get_d_idx_map(beta_lst, gamma_lst_c):
    assert len(beta_lst) == gamma_lst_c.shape[0]
    d_num = len(BETA_RANGE_LST)
    for c in range(C):
        d_num *= len(GAMMA_RANGE_DCT[c + 1])
    d_idx_start_lst = np.zeros(d_num + 1)
    beta_gamma_range_lst = get_beta_gamma_range_lst()
    assert d_num == len(beta_gamma_range_lst)
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

def get_d_idx_map_v2(beta_lst, gamma_lst_c):
    assert len(beta_lst) == gamma_lst_c.shape[0]
    d_num = len(BETA_GAMMA_RANGE_LST)
    d_idx_start_lst = np.zeros(d_num + 1)
    beta_gamma_range_lst = get_beta_gamma_range_lst()
    assert d_num == len(beta_gamma_range_lst)
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

def get_grid_shorter():
    beta_vec = np.linspace(BETA_RANGE[0], BETA_RANGE[1], INT_GRID + 1)
    gamma_mat = np.zeros((C, INT_GRID + 1))
    for c in range(1, C):
        gamma_c_grid = np.linspace(GAMMA_RANGE_C[c][0], GAMMA_RANGE_C[c][1], INT_GRID + 1)
        gamma_mat[c,:] = gamma_c_grid
    beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
    gamma_mat = (gamma_mat[:,1:] + gamma_mat[:,:-1]) / 2
    beta_gamma_range_lst = get_beta_gamma_range_lst()
    beta_lst = []
    gamma_lst_c = []
    for lst in beta_gamma_range_lst:
        beta_curr = beta_vec[(beta_vec > lst[0][0]) & (beta_vec <= lst[0][1])]
        gamma_c_curr = []
        for c in range(1, C):
            tmp = gamma_mat[c,:][(gamma_mat[c,:] > lst[c+1][0]) & (gamma_mat[c,:] <= lst[c+1][1])]
            gamma_c_curr.append(tmp)
        grid_tup = [x.ravel() for x in np.meshgrid(beta_curr, *gamma_c_curr, indexing = "ij")]
        beta_lst_curr = grid_tup[0]
        gamma_lst_c_curr = np.vstack(grid_tup[1:]).T
        gamma_lst_c_curr = np.hstack((np.zeros((gamma_lst_c_curr.shape[0], 1)), gamma_lst_c_curr))
        beta_lst.append(beta_lst_curr)
        gamma_lst_c.append(gamma_lst_c_curr)
    beta_lst = np.concatenate(beta_lst)
    gamma_lst_c = np.concatenate(gamma_lst_c)
    d_idx_start_lst = get_d_idx_map(beta_lst, gamma_lst_c)
    gamma_lst_c = gamma_lst_c.cumsum(axis = 1)
    return beta_lst, gamma_lst_c, d_idx_start_lst

def get_grid(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT):
#    beta_vec = np.linspace(BETA_RANGE[0], BETA_RANGE[1], INT_GRID + 1)
#    gamma_mat = np.zeros((C, INT_GRID + 1))
#    for c in range(1, C):
#        gamma_c_grid = np.linspace(GAMMA_RANGE_C[c][0], GAMMA_RANGE_C[c][1], INT_GRID + 1)
#        gamma_mat[c,:] = gamma_c_grid
#    beta_vec = (beta_vec[1:] + beta_vec[:-1]) / 2
#    gamma_mat = (gamma_mat[:,1:] + gamma_mat[:,:-1]) / 2
    beta_gamma_range_lst = get_beta_gamma_range_lst(beta_range_lst, gamma_range_dct)
    beta_lst = []
    gamma_lst_c = []
    C = len(list(gamma_range_dct.keys()))
    for lst in beta_gamma_range_lst:
        beta_curr = np.linspace(lst[0][0], lst[0][1], INT_GRID + 1) #beta_vec[(beta_vec > lst[0][0]) & (beta_vec <= lst[0][1])]
        gamma_c_curr = []
        for c in range(1, C):
            tmp = np.linspace(lst[c+1][0], lst[c+1][1], INT_GRID + 1) #gamma_mat[c,:][(gamma_mat[c,:] > lst[c+1][0]) & (gamma_mat[c,:] <= lst[c+1][1])]
            gamma_c_curr.append(tmp)
        grid_tup = [x.ravel() for x in np.meshgrid(beta_curr, *gamma_c_curr, indexing = "ij")]
        beta_lst_curr = grid_tup[0]
        gamma_lst_c_curr = np.vstack(grid_tup[1:]).T
        gamma_lst_c_curr = np.hstack((np.zeros((gamma_lst_c_curr.shape[0], 1)), gamma_lst_c_curr))
        beta_lst.append(beta_lst_curr)
        gamma_lst_c.append(gamma_lst_c_curr)
    beta_lst = np.concatenate(beta_lst)
    gamma_lst_c = np.concatenate(gamma_lst_c)
    d_idx_start_lst = get_d_idx_map(beta_lst, gamma_lst_c)
    gamma_lst_c = gamma_lst_c.cumsum(axis = 1)
    return beta_lst, gamma_lst_c, d_idx_start_lst

def profile_given_data_single(lo, hi, beta_lst, gamma_lst_c, segment_type_num):
    sigma_ns_h = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    sigma_ns_o = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    for data_idx in tqdm(range(lo, hi)):
        sigma_s_h, sigma_s_o = solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, LATENCY_O_LST[data_idx,:], LATENCY_HOV_LST[data_idx,:], TAU_CS_LST[data_idx,:,:])
        sigma_ns_h[data_idx,:,:,:,:] = sigma_s_h[0,:,:,:,:]
        sigma_ns_o[data_idx,:,:,:,:] = sigma_s_o[0,:,:,:,:]
    return sigma_ns_h, sigma_ns_o

def get_d_coef_matrix(sigma_ns_h, sigma_ns_o):
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    segment_type_num = int(S * (S + 1) / 2)
    ## Compute equilibrium flow using d
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    ### Compute equilibrium flows
    ## TODO: Implement d_to_f_mat
    ### o + h
    d_coef_matrix = np.zeros((2 * N_DATA, d_len))
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
                            d_coef_matrix[relev_data_idx, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num
                            d_coef_matrix[N_DATA + relev_data_idx, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num
                    segment_idx += 1
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

def is_identifiable(sigma_ns_h, sigma_ns_o):
    d_coef_matrix = get_d_coef_matrix(sigma_ns_h, sigma_ns_o)
    mat_rank = np.linalg.matrix_rank(d_coef_matrix)
    print(mat_rank, d_coef_matrix.shape)
#    d_coef_matrix_shorter, d_idx_dropped = drop_dependent_columns(d_coef_matrix)
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

def generate_density(hourly_demand_weights = [], segment_demand_lst = [], density_lst = [], beta_range_lst = [], gamma_range_dct = {}, save = True, name = ""):
    ### Get grid
    n_hours = len(hourly_demand_weights)
    segment_type_num = len(segment_demand_lst)
    n_segments = None
    meta_data = {
        "N_HOUR": n_hours,
        "S": n_segments,
        "C": len(list(gamma_range_dct.keys())),
        "segment_type_num": segment_type_num,
        "BETA_RANGE_LST": beta_range_lst,
        "GAMMA_RANGE_DCT": gamma_range_dct
    }
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid(beta_range_lst, gamma_range_dct)
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    density = np.zeros(d_len)
    for hour_idx in range(n_hours):
        for d_idx in range(single_t_d_len):
            elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
            segment_idx = 0
            for s_o in range(n_segments):
                for s_d in range(s_o, n_segments):
                    density_idx = hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx
                    density_val = hourly_demand_weights[hour_idx] * segment_demand_lst[segment_idx] * density_lst[d_idx]
                    density[density_idx] = density_val
                    segment_idx += 1
    if save:
        np.save(f"density/preference_density_synthetic_{name}.npy", density)
        with open(f"density/preference_density_synthetic_{name}_meta.json", "w") as json_file:
            json.dump(d, json_file)
    return density, meta_data

def calibrate_density_synthetic(meta_data = None, data_dct = None):
    if meta_data is not None:
        N_HOUR = meta_data["N_HOUR"]
        S = meta_data["S"]
        C = meta_data["C"]
        BETA_RANGE_LST = meta_data["BETA_RANGE_LST"]
        GAMMA_RANGE_DCT = meta_data["GAMMA_RANGE_DCT"]
    if data_dct is not None:
        N_DATA
        TRAIN_IDX
        FLOW_O_TARGET
        FLOW_H_TARGET
        FLOW_COEF
        LANE_TYPE_ALL
        SEGMENT_LST_ALL
        HOUR_LST_ALL
    ## Get sigma profile for each grid
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT)
    segment_type_num = int(S * (S + 1) / 2)
    ### Compute profile given data
    sigma_ns_h = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    sigma_ns_o = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    batch_size = int(math.ceil(N_DATA / N_CPU))
    results = Parallel(n_jobs = N_CPU)(delayed(profile_given_data_single)(
        i * batch_size, min(N_DATA, (i + 1) * batch_size), beta_lst, gamma_lst_c, segment_type_num
    ) for i in range(N_CPU))
    for res in tqdm(results):
        sigma_ns_h += res[0]
        sigma_ns_o += res[1]
    d_idx_dropped = is_identifiable(sigma_ns_h, sigma_ns_o)
    ## Compute equilibrium flow using d
    model = gp.Model()
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    d = model.addMVar(d_len, lb = 0, vtype = GRB.CONTINUOUS, name = "d")
    ### Compute equilibrium flows
    f_o_equi = model.addMVar(N_DATA * S, lb = 0, vtype = GRB.CONTINUOUS, name = "f_o")
    f_h_equi = model.addMVar(N_DATA * S * C, lb = 0, vtype = GRB.CONTINUOUS, name = "fh")
    ## TODO: Implement d_to_f_mat
    ### o + h
    d_to_fo_mat = np.zeros((N_DATA * S, d_len))
    d_to_fh_mat = np.zeros((C * N_DATA * S, d_len))
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
                            d_to_fo_mat[relev_data_idx * S + s, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num #/ C #/ (s_d - s_o + 1)
                            d_to_fh_mat[relev_data_idx * S * C  + s * C + c, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num #/ C #/ (s_d - s_o + 1)
                    segment_idx += 1
    model.addConstr(d_to_fo_mat @ d == f_o_equi)
    model.addConstr(d_to_fh_mat @ d == f_h_equi)
    ### Compute objective function
    objective = ((f_o_equi[:(TRAIN_IDX * S)] - FLOW_O_TARGET[:(TRAIN_IDX * S)]) * FLOW_COEF * (f_o_equi[:(TRAIN_IDX * S)] - FLOW_O_TARGET[:(TRAIN_IDX * S)]) * FLOW_COEF).sum() / TRAIN_IDX
    objective += ((f_h_equi[:(TRAIN_IDX * S * C)] - FLOW_H_TARGET[:(TRAIN_IDX * S * C)]) * (f_h_equi[:(TRAIN_IDX * S * C)] - FLOW_H_TARGET[:(TRAIN_IDX * S * C)])).sum() / TRAIN_IDX
    ### Optimize the model
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    obj_val = model.ObjVal
    density = np.zeros(d_len)
    for i in range(d_len):
        density[i] = d[i].x
    f_o_equi_ret = d_to_fo_mat @ density
    f_h_equi_ret = d_to_fh_mat @ density
    df_tmp = pd.DataFrame.from_dict({"Flow Equi": f_equi_ret, "Flow Target": FLOW_TARGET})
    df_tmp["Lane Type"] = LANE_TYPE_ALL
    df_tmp["Hour"] = HOUR_LST_ALL
    df_tmp["Segment"] = SEGMENT_LST_ALL
    df_tmp.to_csv("tmp.csv", index = False)
    return density

def calibrate_density(meta_data = None):
    if meta_data is not None:
        N_HOUR = meta_data["N_HOUR"]
        S = meta_data["S"]
        C = meta_data["C"]
        BETA_RANGE_LST = meta_data["BETA_RANGE_LST"]
        GAMMA_RANGE_DCT = meta_data["GAMMA_RANGE_DCT"]
    ## Get sigma profile for each grid
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid(beta_range_lst = BETA_RANGE_LST, gamma_range_dct = GAMMA_RANGE_DCT)
    segment_type_num = int(S * (S + 1) / 2)
    ### Compute profile given data
    sigma_ns_h = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    sigma_ns_o = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    batch_size = int(math.ceil(N_DATA / N_CPU))
    results = Parallel(n_jobs = N_CPU)(delayed(profile_given_data_single)(
        i * batch_size, min(N_DATA, (i + 1) * batch_size), beta_lst, gamma_lst_c, segment_type_num
    ) for i in range(N_CPU))
    for res in tqdm(results):
        sigma_ns_h += res[0]
        sigma_ns_o += res[1]
    d_idx_dropped = is_identifiable(sigma_ns_h, sigma_ns_o)
    ## Compute equilibrium flow using d
    model = gp.Model()
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    d = model.addMVar(d_len, lb = 0, vtype = GRB.CONTINUOUS, name = "d")
    ### Compute equilibrium flows
    f_equi = model.addMVar(2 * N_DATA * S, lb = 0, vtype = GRB.CONTINUOUS, name = "f")
    f_h_equi = model.addMVar(C * N_DATA, lb = 0, vtype = GRB.CONTINUOUS, name = "fh")
    f_h_total_equi = model.addMVar(N_DATA, lb = 0, vtype = GRB.CONTINUOUS, name = "fh_total")
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
                            d_to_f_mat[relev_data_idx * S + s, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num #/ C #/ (s_d - s_o + 1)
                            d_to_f_mat[N_DATA * S + relev_data_idx * S + s, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num #/ C #/ (s_d - s_o + 1)
                            d_to_fh_mat[c * N_DATA + relev_data_idx, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num #/ C #/ (s_d - s_o + 1)
                            d_to_fh_total_mat[relev_data_idx, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num #/ C #/ (s_d - s_o + 1)
                    segment_idx += 1
    model.addConstr(d_to_f_mat @ d == f_equi)
    model.addConstr(d_to_fh_mat @ d == f_h_equi)
    model.addConstr(d_to_fh_total_mat @ d == f_h_total_equi)
    ### Compute objective function
    objective = ((f_equi[:(2 * TRAIN_IDX * S)] - FLOW_TARGET[:(2 * TRAIN_IDX * S)]) * FLOW_COEF[:(2 * TRAIN_IDX * S)] * (f_equi[:(2 * TRAIN_IDX * S)] - FLOW_TARGET[:(2 * TRAIN_IDX * S)]) * FLOW_COEF[:(2 * TRAIN_IDX * S)]).sum() / TRAIN_IDX
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
    density = np.zeros(d_len)
    for i in range(d_len):
        density[i] = d[i].x
    f_equi_ret = d_to_f_mat @ density
    df_tmp = pd.DataFrame.from_dict({"Flow Equi": f_equi_ret, "Flow Target": FLOW_TARGET})
    df_tmp["Lane Type"] = LANE_TYPE_ALL
    df_tmp["Hour"] = HOUR_LST_ALL
    df_tmp["Segment"] = SEGMENT_LST_ALL
    df_tmp.to_csv("tmp.csv", index = False)
    dct_ratio = {"Date": date_lst}
    f_h_ret = np.zeros(C * N_DATA)
    for i in range(C * N_DATA):
        f_h_ret[i] = f_h_equi[i].x
    f_h_total_ret = np.zeros(N_DATA)
    for i in range(N_DATA):
        f_h_total_ret[i] = f_h_total_equi[i].x
    flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_ret
    for c in range(C):
        dct_ratio[f"Equi {c}"] = PROFILE_DATE_MAP @ f_h_ret[(c*N_DATA):((c+1)*N_DATA)]
        dct_ratio[f"Target {c}"] = RATIO_TARGET[:,c] * flow_ratio_target_total
    df_tmp_ratio = pd.DataFrame.from_dict(dct_ratio)
    df_tmp_ratio.to_csv("tmp_ratio.csv", index = False)
    return density

def describe_density(density):
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    beta_gamma_range_lst = get_beta_gamma_range_lst()
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
                val = density[hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx]
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

def get_flow_from_toll_iterative(density, tau_cs, rho = 0.25, hour_idx = 12, num_itr = 10, lam = 0.5):
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    n_grids = len(beta_lst)
    segment_type_num = int(S * (S + 1) / 2)
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    ### Compute auxiliary matrices
    segment_pop = get_segment_pop(density, hour_idx)
    segment_type_strategy_len = segment_type_num * C * S * 2
    equi_profile_len = len(beta_lst) * segment_type_num * C * S * 2
    segment_type_strategy_to_flow_o_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_flow_h_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_agents_o_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_agents_h_map = np.zeros((S, segment_type_strategy_len))
    equi_profile_to_strategy_density_vec = np.zeros((len(beta_lst), segment_type_strategy_len))
    equi_profile_to_strategy_pop_vec = np.zeros((len(beta_lst), segment_type_strategy_len))
    segment_len_lst = np.zeros(segment_type_num)
    for c in range(C):
        segment_type_idx = 0
        for s_o in range(S):
            for s_d in range(s_o, S):
                col_idx_o_begin = segment_type_idx * C * S * 2 + c * S * 2 + s_o * 2
                col_idx_o_end = segment_type_idx * C * S * 2 + c * S * 2 + (s_d + 1) * 2
                col_idx_h_begin = col_idx_o_begin + 1
                col_idx_h_end = col_idx_o_end + 1
                segment_type_strategy_to_flow_o_map[s_o:(s_d+1), col_idx_o_begin:col_idx_o_end:2] = 1 / (c + 1) * segment_pop[segment_type_idx]
                segment_type_strategy_to_flow_h_map[s_o:(s_d+1), col_idx_h_begin:col_idx_h_end:2] = 1 / (c + 1) * segment_pop[segment_type_idx]
                segment_type_strategy_to_agents_o_map[s_o:(s_d+1), col_idx_o_begin:col_idx_o_end:2] = segment_pop[segment_type_idx]
                segment_type_strategy_to_agents_h_map[s_o:(s_d+1), col_idx_h_begin:col_idx_h_end:2] = segment_pop[segment_type_idx]
                segment_len_lst[segment_type_idx] = s_d + 1 - s_o
                segment_type_idx += 1
    segment_density_lst = np.zeros(segment_type_num)
    ### Begin solving strategy profile iteratively
    loss_arr = []
    ### Guess a strategy profile
    segment_type_strategy = np.zeros(segment_type_strategy_len)
    ### TODO: Mask out infeasible S of each segment type
    segment_type_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            density_sum = density[(hour_idx * single_t_d_len * segment_type_num + segment_type_idx):((hour_idx + 1) * single_t_d_len * segment_type_num):segment_type_num].sum()
            segment_density_lst[segment_type_idx] = density_sum
            seg_start = segment_type_idx * C * S * 2
            seg_end = (segment_type_idx + 1) * C * S * 2
            for s in range(s_o, s_d + 1):
                seg_start = segment_type_idx * C * S * 2
                begin = seg_start + s * 2
                end = seg_start + C * S * 2
                seg_end = (segment_type_idx + 1) * C * S * 2
                o_idx_lst = np.arange(begin, seg_end, S * 2)
                h_idx_lst = o_idx_lst + 1
                if segment_density_lst[segment_type_idx] > 1:
                    segment_type_strategy[o_idx_lst] = 1 / (segment_len_lst[segment_type_idx] * C * 2)
                    segment_type_strategy[h_idx_lst] = 1 / (segment_len_lst[segment_type_idx] * C * 2)
                    if density_sum > 1:
                        for d_idx in range(single_t_d_len):
                            d_val = density[hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_type_idx]
                            elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
                            equi_val = d_val / elem_num / density_sum / segment_len_lst[segment_type_idx]
                            equi_profile_to_strategy_density_vec[d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1],o_idx_lst] = equi_val
                            equi_profile_to_strategy_density_vec[d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1],h_idx_lst] = equi_val
                            equi_profile_to_strategy_pop_vec[d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1],o_idx_lst] = equi_val * density_sum
                            equi_profile_to_strategy_pop_vec[d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1],h_idx_lst] = equi_val * density_sum
            segment_type_idx += 1
#    print(equi_profile_to_strategy_density_vec.sum(), segment_type_strategy.sum())
    o_lanes = int(NUM_LANES * (1 - rho))
    h_lanes = NUM_LANES - o_lanes
    utility_cost_arr = []
    tau_lst = np.zeros((1, segment_type_strategy_len))
    tau_lst[:,1::2] = np.tile(tau_cs.reshape(C * S), segment_type_num)
    gamma_lst_c_long = np.tile(gamma_lst_c.repeat(S * 2, axis = 1), reps = (1, segment_type_num))
    for itr in tqdm(range(num_itr), leave = False):
        ### Compute the corresponding latency
        flow_o = segment_type_strategy_to_flow_o_map @ segment_type_strategy
        flow_h = segment_type_strategy_to_flow_h_map @ segment_type_strategy
        latency_o = get_cost(flow_o / o_lanes, DISTANCE_ARR)
        latency_h = get_cost(flow_h / h_lanes, DISTANCE_ARR)
        ### Solve the equilibrium profile
        sigma_s_h, sigma_s_o = solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, latency_o, latency_h, tau_cs)
        sigma_s = np.zeros((len(beta_lst), segment_type_strategy_len))
        sigma_s[:,::2] = sigma_s_o.reshape((len(beta_lst), segment_type_strategy_len // 2))
        sigma_s[:,1::2] = sigma_s_h.reshape((len(beta_lst), segment_type_strategy_len // 2))
        equi_profile = (equi_profile_to_strategy_density_vec * sigma_s).sum(axis = 0)
        ### Update the guess
        loss = np.mean((segment_type_strategy - equi_profile) ** 2)
        segment_type_strategy = segment_type_strategy * (1 - lam) + equi_profile * lam
        loss_arr.append(loss)
        latency_tmp = np.zeros(S * 2)
        latency_tmp[::2] = latency_o
        latency_tmp[1::2] = latency_h
        latency_lst = np.tile(latency_tmp, segment_type_num * C).reshape((1, segment_type_strategy_len))
        if itr > 0:
            total_utility_cost_prev = (equi_profile_dens * (beta_lst.reshape((len(beta_lst), 1)) * latency_lst + tau_lst + gamma_lst_c_long)).sum()
            equi_profile_dens = equi_profile_to_strategy_density_vec * sigma_s
            total_utility_cost = (equi_profile_dens * (beta_lst.reshape((len(beta_lst), 1)) * latency_lst + tau_lst + gamma_lst_c_long)).sum()
            utility_cost_arr.append(total_utility_cost_prev - total_utility_cost)
        else:
            equi_profile_dens = equi_profile_to_strategy_density_vec * sigma_s
        
    flow_o = segment_type_strategy_to_flow_o_map @ segment_type_strategy
    flow_h = segment_type_strategy_to_flow_h_map @ segment_type_strategy
    latency_o = get_cost(flow_o / o_lanes, DISTANCE_ARR)
    latency_h = get_cost(flow_h / h_lanes, DISTANCE_ARR)
    print("Ordinary Flow:", flow_o)
    print("HOT Flow:", flow_h)
    print("Ordinary Travel Time:", latency_o)
    print("HOT Travel Time:", latency_h)
    equi_profile_pop = equi_profile_to_strategy_pop_vec * sigma_s
    agents_o = segment_type_strategy_to_agents_o_map @ segment_type_strategy
    agents_h = segment_type_strategy_to_agents_h_map @ segment_type_strategy
    total_travel_time = (agents_o * latency_o + agents_h * latency_h).sum()
    total_emission = (flow_o * latency_o + flow_h * latency_h).sum()
    total_revenue = (equi_profile_pop * tau_lst).sum()
    latency_tmp = np.zeros(S * 2)
    latency_tmp[::2] = latency_o
    latency_tmp[1::2] = latency_h
    latency_lst = np.tile(latency_tmp, segment_type_num * C).reshape((1, segment_type_strategy_len))
    total_utility_cost = (equi_profile_pop * (beta_lst.reshape((len(beta_lst), 1)) * latency_lst + tau_lst + gamma_lst_c_long)).sum()
    return segment_type_strategy, loss_arr, utility_cost_arr, latency_o, latency_h, total_travel_time, total_emission, total_revenue, total_utility_cost

def describe_segment_type_strategy(sigma, density, hour_idx, eps = 1e-3):
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    segment_range_lst = []
    segment_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            name = f"{segment_lst[s_o]} to {segment_lst[s_d]}"
            segment_range_lst.append(name)
            segment_idx += 1
    # len(beta_lst) * segment_type_num * C * S * 2
    segment_type_num = int(S * (S + 1) / 2)
    segment_pop = get_segment_pop(density, hour_idx)
    for segment_idx in range(segment_type_num):
        print(f"Segment {segment_range_lst[segment_idx]}:")
        pop = segment_pop[segment_idx]
        for s in range(S):
            for c in range(C):
                sigma_o_idx = np.arange(segment_idx * C * S * 2 + c * S * 2 + s * 2, len(sigma), segment_type_num * C * S * 2)
#                denom = (d_vec[:,segment_idx] / d_total[segment_idx]).sum()
                sigma_o_total = (sigma[sigma_o_idx] * 1).sum() / 1
                sigma_h_total = (sigma[sigma_o_idx + 1] * 1).sum() / 1
                if sigma_o_total + sigma_h_total > eps:
                    print(f"\tS = {s}, C = {c + 1}: sigma_o = {sigma_o_total:.2f}, sigma_h = {sigma_h_total:.2f}")

def toll_design_grid_search_single(tau_tup_lst, density, hour_idx = 12, tau_max = 5, d_tau = 1, rho_lst = [0.25, 0.5, 0.75], num_itr = 1000, lam = 1e-2):
    dct_results = {"Rho": [], "Loss": [], "Total Travel Time": [], "Total Emission": [], "Total Revenue": [], "Total Utility Cost": []}
    for s in range(S):
        dct_results[f"Toll {s}"] = []
    for tau_tup in tqdm(tau_tup_lst):
        ### Currently only support C = 3
        tau_cs = np.zeros((C, S))
        tau_cs[0,:] = np.array(tau_tup)
        tau_cs[1,:] = tau_cs[0,:] / 4
        for rho in rho_lst:
            ### segment_type_num * C * S * 2
            segment_type_strategy, loss_arr, latency_o, latency_h, total_travel_time, total_emission, total_revenue, total_utility_cost = get_flow_from_toll_iterative(density, tau_cs = tau_cs, rho = rho, hour_idx = hour_idx, num_itr = num_itr, lam = lam)
            ### Store results
            dct_results["Rho"].append(rho)
            dct_results["Loss"].append(loss_arr[-1])
            dct_results["Total Travel Time"].append(total_travel_time)
            dct_results["Total Emission"].append(total_emission)
            dct_results["Total Revenue"].append(total_revenue)
            dct_results["Total Utility Cost"].append(total_utility_cost)
            for s in range(S):
                dct_results[f"Toll {s}"].append(tau_tup[s])
    return dct_results

def toll_design_grid_search(density, hour_idx = 12, tau_max = 5, d_tau = 1, rho_lst = [0.25, 0.5, 0.75], num_itr = 1000, lam = 1e-2):
    dct_results = None
    tau_lst_single = np.linspace(0, tau_max, int(tau_max // d_tau) + 1)
    tau_tup_lst = list(itertools.product(*[tau_lst_single]*S))
    batch_size = int(math.ceil(len(tau_tup_lst) / N_CPU))
    results = Parallel(n_jobs = N_CPU)(delayed(toll_design_grid_search_single)(
        tau_tup_lst[(i * batch_size):min((i + 1) * batch_size, len(tau_tup_lst))], density, hour_idx, tau_max, d_tau, rho_lst, num_itr, lam
    ) for i in range(N_CPU))
    for res in results:
        if dct_results is None:
            dct_results = res
        else:
            for key in dct_results:
                dct_results[key] += res[key]
    df = pd.DataFrame.from_dict(dct_results)
    return df

## TODO: Create datasets
hourly_demand_weights = [1, 2]
segment_demand_lst = [1000, 1000, 1000]
density_lst = [1/8] * 8
beta_range_lst = [(0, 1), (1, 2)]
gamma_range_dct = {
    1: [(0, 0)],
    2: [(0, 1), (1, 2)],
    3: [(0, 1), (1, 2)]
}
name = "2hour_3seg_uniform"
generate_density(hourly_demand_weights = hourly_demand_weights, segment_demand_lst = segment_demand_lst, density_lst = density_lst, beta_range_lst = beta_range_lst, gamma_range_dct = gamma_range_dct, save = True, name = name)

if DENSITY_RECALIBRATE:
    density = calibrate_density()
    np.save("density/preference_density_general.npy", density)
else:
    density = np.load("density/preference_density_general.npy")
describe_density(density)

assert False

segment_type_strategy, loss_arr, utility_cost_arr, latency_o, latency_h, total_travel_time, total_emission, total_revenue, total_utility_cost = get_flow_from_toll_iterative(density, tau_cs = np.array([[5, 1.25, 0], [5, 1.25, 0], [5, 1.25, 0], [5, 1.25, 0], [5, 1.25, 0]]).T, rho = 0.25, hour_idx = 12, num_itr = 100, lam = 1e-1)
#print(segment_type_strategy.round(3))
#print(segment_type_strategy.sum())
print(total_travel_time, total_emission, total_revenue, total_utility_cost)
print("Final Loss:", loss_arr[-1])
describe_segment_type_strategy(segment_type_strategy, density, hour_idx = 12, eps = 1e-2)

plt.plot(loss_arr)
#plt.yscale("log")
plt.title(f"loss = {loss_arr[-1]:.2e}")
plt.savefig("loss.png")
plt.clf()
plt.close()

plt.plot(utility_cost_arr)
#plt.yscale("log")
plt.title(f"Utility Cost = {utility_cost_arr[-1]:.2e}")
plt.savefig("utility_cost.png")
plt.clf()
plt.close()

df_all = None
for hour_idx in tqdm(range(15)):
    df = toll_design_grid_search(density, hour_idx = hour_idx, tau_max = 5, d_tau = 1, rho_lst = [0.25], num_itr = 300, lam = 1e-2)
    df["Hour"] = hour_idx + 5
    if df_all is None:
        df_all = df
    else:
        df_all = pd.concat([df_all, df], ignore_index = True)
# print(df)
df_all.to_csv("toll_design_multiseg.csv", index = False)
