import math
import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, csr_array, dia_matrix, vstack
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from joblib import Parallel, delayed
from tqdm import tqdm

BETA_RANGE_LST = [(0, 1), (1, 2)]
GAMMA_RANGE_DCT = {
    1: [(0, 0)],
    2: [(0, 1), (1, 2), (2, 3), (3, 4)],
    3: [(0, 1), (1, 2)]
}
C = 3
BETA_RANGE = (BETA_RANGE_LST[0][0], BETA_RANGE_LST[-1][1])
GAMMA_RANGE_C = [(GAMMA_RANGE_DCT[c][0][0], GAMMA_RANGE_DCT[c][-1][1]) for c in range(1, C + 1)]
INT_GRID = 10 #50
## Load Data
### Date, Hour, Segment, HOV Flow, Ordinary Flow, HOV Travel Time, Ordinary Travel Time, Avg_total_toll
df = pd.read_csv("data/df_meta.csv") #pd.read_csv("hourly_demand_20210401.csv")
# df = df[df["Segment"] == "3460 - Hesperian/238 NB"]
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df = df.dropna()
df = df[(df["Date"] >= "2021-03-01") & (df["Date"] <= "2021-08-31")]

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
HOUR_LST = np.array(df_wide["Hour"])
N_HOUR = len(df_wide["Hour"].unique())
UNIQUE_HOUR_LST = np.array(df["Hour"].unique())

density = np.load("density/preference_density_general.npy")

def elem_in_range(beta, gamma_c, lst):
    if beta > lst[0][1]:
        return False
    for c in range(C):
        if gamma_c[c] > lst[c + 1][1]:
            return False
    return True

def get_beta_gamma_range_lst():
    beta_gamma_range_lst = [[x] for x in BETA_RANGE_LST]
    for c in range(C):
        tmp = []
        for lst in beta_gamma_range_lst:
            for tup in GAMMA_RANGE_DCT[c + 1]:
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

def get_grid():
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
    segment_pop_aggr = np.zeros(S)
    for s in range(S):
        segment_type_idx_begin = s * (2 * S + 1 - s) // 2
        segment_type_idx_end = (s + 1) * (2 * S - s) // 2
        segment_pop_aggr[s] = segment_pop[segment_type_idx_begin:segment_type_idx_end].sum()
    return segment_pop, segment_pop_aggr

def get_pref_dens(density, hour_idx, segment_idx, carpool_num = 2):
    assert carpool_num in [2, 3]
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    density_idx_begin = hour_idx * single_t_d_len * segment_type_num + 0
    density_idx_end = (hour_idx + 1) * single_t_d_len * segment_type_num
    pref_dens = np.zeros(single_t_d_len)
    segment_type_idx_begin = segment_idx * (2 * S + 1 - segment_idx) // 2
    segment_type_idx_end = (segment_idx + 1) * (2 * S - segment_idx) // 2
    for relv_segment_type_idx in range(segment_type_idx_begin, segment_type_idx_end):
        pref_dens += density[(density_idx_begin + relv_segment_type_idx):density_idx_end:segment_type_num]
    print(pref_dens.round())

#describe_density(density)

segment_pop, segment_pop_aggr = get_segment_pop(density, 12)
print(segment_pop_aggr.round())

#get_pref_dens(density, 12, 0, carpool_num = 2)
