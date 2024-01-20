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
from tqdm import tqdm

## Script Options
DENSITY_RECALIBRATE = True

## Hyperparameters
NUM_LANES = 4
BPR_POWER = 4
BPR_A = 7e-4 #2.4115e-13
BPR_B = 0.7906
DISTANCE = 7.16 # miles

BETA_RANGE_LST = [(0, 1), (1, 2)]
GAMMA_RANGE_DCT = {
    1: [(0, 0)],
    2: [(0, 1), (1, 2), (2, 3), (3, 4)],
    3: [(0, 1), (1, 2)]
}
C = 3
S = 1
BETA_RANGE = (BETA_RANGE_LST[0][0], BETA_RANGE_LST[-1][1])
GAMMA_RANGE_C = [(GAMMA_RANGE_DCT[c][0][0], GAMMA_RANGE_DCT[c][-1][1]) for c in range(1, C + 1)]
INT_GRID = 50

## Load Data
### Date, Hour, Segment, HOV Flow, Ordinary Flow, HOV Travel Time, Ordinary Travel Time, Avg_total_toll
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
df_pop["Sigma_1ratio"] = df_pop["Single"] / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_2ratio"] = df_pop["TwoPeople"] * 2 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df_pop["Sigma_3ratio"] = df_pop["ThreePlus"] * 3 / (df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3)
df = df.merge(df_pop[["Date", "Sigma_1ratio", "Sigma_2ratio", "Sigma_3ratio"]], on = "Date")
df = df.sort_values(["Date", "Hour"], ascending = True)
TAU_LST = np.array(df["Avg_total_toll"]) #list(df["Toll"])
TAU_CS_LST = np.zeros((df.shape[0], C, S))
N_DATA = df.shape[0] #100#
df = df.iloc[:N_DATA]
HOUR_LST = np.array(df["Hour"])
N_HOUR = len(df["Hour"].unique())
UNIQUE_HOUR_LST = np.array(df["Hour"].unique())
### TODO: Change it to multisegments later
TAU_CS_LST[:,0,0] = TAU_LST
TAU_CS_LST[:,1,0] = TAU_CS_LST[:,0,0] / 4
LATENCY_O_LST = np.array(df["Ordinary Travel Time"]).reshape((N_DATA, 1))
LATENCY_HOV_LST = np.array(df["HOV Travel Time"]).reshape((N_DATA, 1))
FLOW_O_LST = np.array(df["Ordinary Flow"])
FLOW_HOV_LST = np.array(df["HOV Flow"])
FLOW_TARGET = np.concatenate((FLOW_O_LST, FLOW_HOV_LST))
###
N_DATES = len(df["Date"].unique())
## N_DATES, N_DATA, S
## Days to ignore: 3/31, 4/23, 4/26, 6/30
RATIO_INDEX_TO_IGNORE = [22, 39, 40, 86]
PROFILE_DATE_MAP = np.zeros((N_DATES - len(RATIO_INDEX_TO_IGNORE), N_DATA))
RATIO_TARGET = np.zeros((N_DATES - len(RATIO_INDEX_TO_IGNORE), C))
date_lst = list(df.drop_duplicates("Date")["Date"])
idx = 0
tmp = []
for i in range(len(date_lst)):
    date = date_lst[i]
    sigma_1ratio = df[df["Date"] == date].iloc[0]["Sigma_1ratio"]
    sigma_2ratio = df[df["Date"] == date].iloc[0]["Sigma_2ratio"]
    sigma_3ratio = df[df["Date"] == date].iloc[0]["Sigma_3ratio"]
    idx_lst = np.array(df[df["Date"] == date].index)
    if i not in RATIO_INDEX_TO_IGNORE:
        PROFILE_DATE_MAP[idx, idx_lst] = 1
        RATIO_TARGET[idx, 0] = sigma_1ratio
        RATIO_TARGET[idx, 1] = sigma_2ratio
        RATIO_TARGET[idx, 2] = sigma_3ratio
        idx += 1
        tmp.append(date)
date_lst = tmp

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
    gamma_lst_c = gamma_lst_c.reshape((1, n_grids, C, S))
    n_data = 1#len(c_o)
#    c_o = c_o.reshape((n_data, 1, 1, 1))
#    c_h = c_h.reshape((n_data, 1, 1, 1))
    tau_cs = tau_cs.reshape((n_data, 1, C, S))
    cost_o = beta_lst * c_o
    cost_h = beta_lst * c_h + gamma_lst_c + tau_cs
    lane_cs = (cost_h < c_o) + 0
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
    lane_cs_ret = np.zeros((n_data, n_grids, segment_type_num, C, S))
    for data_idx in tqdm(range(n_data), leave = False):
        for grid_idx in tqdm(range(n_grids), leave = False):
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    best_c = best_c_lst[segment_idx][data_idx, grid_idx]
                    lane_cs_ret[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    segment_idx += 1
    return lane_cs_ret #lane_cs[:,best_c,:]

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

def calibrate_density():
    ## Get sigma profile for each grid
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    segment_type_num = int(S * (S + 1) / 2)
    ### Compute profile given data
    sigma_ns = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S))
    for data_idx in tqdm(range(N_DATA)):
        sigma_s = solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, LATENCY_O_LST[data_idx], LATENCY_HOV_LST[data_idx], TAU_CS_LST[data_idx,:,:])
#        print(data_idx, ":")
#        for c in range(C):
#            print("\t", c, sigma_s[0,:,c,:].mean() / (c + 1))
        sigma_ns[data_idx,:,:,:,:] = sigma_s[0,:,:,:,:]
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
                            d_to_f_mat[relev_data_idx * S + s, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * (1 - sigma_ns[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s]).sum(axis = 1) / elem_num / (s_d - s_o + 1) / C
                            d_to_f_mat[N_DATA * S + relev_data_idx * S + s, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += 1 / (c + 1) * sigma_ns[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num / (s_d - s_o + 1) / C
                            d_to_fh_mat[c * N_DATA + relev_data_idx, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += sigma_ns[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num / (s_d - s_o + 1) / C
                            d_to_fh_total_mat[relev_data_idx, hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx] += sigma_ns[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx+1], segment_idx, c, s].sum(axis = 1) / elem_num / (s_d - s_o + 1) / C
                    segment_idx += 1
    model.addConstr(d_to_f_mat @ d == f_equi)
    model.addConstr(d_to_fh_mat @ d == f_h_equi)
    model.addConstr(d_to_fh_total_mat @ d == f_h_total_equi)
    objective = ((f_equi - FLOW_TARGET) * (f_equi - FLOW_TARGET)).sum() / N_DATA
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
        ratio_loss = (PROFILE_DATE_MAP @ f_h_equi[(c*N_DATA):((c+1)*N_DATA)] - RATIO_TARGET[:,c] * flow_ratio_target_total) #/ N_HOUR
        objective += (ratio_loss * ratio_loss).sum() / N_DATA * 10
#    model.addConstr(ratio_total >= daily_flow_lb)
    objective += ((ratio_total - daily_flow_lb) * (ratio_total - daily_flow_lb)).sum() / N_DATA * 1
    ### Optimize the model
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    obj_val = model.ObjVal
    density = np.zeros(d_len)
    for i in range(d_len):
        density[i] = d[i].x
    f_equi_ret = d_to_f_mat @ density
    df_tmp = pd.DataFrame.from_dict({"Flow Equi": f_equi_ret, "Flow Target": FLOW_TARGET})
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
    single_t_d_len = len(d_idx_start_lst) - 1
    for hour_idx in range(N_HOUR):
        t = UNIQUE_HOUR_LST[hour_idx]
        print(f"Hour = {t}:")
        for segment_idx in range(segment_type_num):
            print(f"\tSegment type = {segment_idx}:")
            for d_idx in range(single_t_d_len):
                val = density[hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx]
                tup = beta_gamma_range_lst[d_idx]
                if val > 1e-3:
                    print(f"\t\tBeta = {tup[0]}, Gamma = {tup[1:]}: {val}")

def solve_equi_sigma_opt(density, hour_idx, rho, tau_tcs):
    ## Initialize
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    ### Initialize the model
    model = gp.Model()
    model.setParam("NonConvex", 2)
    ### Compute sigma to flow maps
    sigma_len = len(beta_lst) * segment_type_num * C * S * 2
    flow_len = S * 2
    sigma_to_flow_map = np.zeros((flow_len, sigma_len))
    ### Compute sigma to cost maps
    sigma_to_latency_coef_map = np.zeros((flow_len, sigma_len))
    sigma_cost_coef = np.zeros(sigma_len)
#    for hour_idx in tqdm(range(N_HOUR)):
    t = UNIQUE_HOUR_LST[hour_idx]
    for d_idx in range(single_t_d_len):
        elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
        segment_idx = 0
        for s_o in range(S):
            for s_d in range(s_o, S):
                d_val = density[hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx]
                for s in range(s_o, s_d + 1):
                    for c in range(C):
                        sigma_idx = np.arange(d_idx_start_lst[d_idx] * segment_type_num * C * S * 2 + segment_idx * C * S * 2 + c * S * 2 + s * 2, d_idx_start_lst[d_idx + 1] * segment_type_num * C * S * 2, segment_type_num * C * S * 2)
                        coef_from_d = d_val / elem_num #/ C / (s_d - s_o + 1)
                        sigma_coef = 1 / (c + 1) * coef_from_d
                        flow_o_idx = 2 * s #hour_idx * S * 2 + s
                        flow_h_idx = 2 * s + 1 #hour_idx * S * 2 + s + 1
                        sigma_to_flow_map[flow_o_idx, sigma_idx] += sigma_coef
                        sigma_to_flow_map[flow_h_idx, sigma_idx + 1] += sigma_coef
                        coef_from_beta = beta_lst[d_idx_start_lst[d_idx] : d_idx_start_lst[d_idx+1]]
                        coef_from_gamma = gamma_lst_c[d_idx_start_lst[d_idx] : d_idx_start_lst[d_idx+1], c]
                        tau = tau_tcs[0,c,s] #tau_tcs[hour_idx, c, s]
                        sigma_to_latency_coef_map[flow_o_idx, sigma_idx] += coef_from_d * coef_from_beta
                        sigma_to_latency_coef_map[flow_h_idx, sigma_idx + 1] += coef_from_d * coef_from_beta
                        sigma_cost_coef[sigma_idx] = coef_from_d * coef_from_gamma
                        sigma_cost_coef[sigma_idx + 1] = coef_from_d * (coef_from_gamma + tau)
                segment_idx += 1
    sigma = model.addMVar(sigma_len, lb = 0, vtype = GRB.CONTINUOUS, name = "sigma")
    flow = model.addMVar(flow_len, lb = 0, vtype = GRB.CONTINUOUS, name = "flow")
    model.addConstr(sigma_to_flow_map @ sigma == flow)
    ### Compute latency from flows
    ###  Currently only support BPR-like cost functions
    flow_power = model.addMVar(flow_len, lb = 0, vtype = GRB.CONTINUOUS, name = "flow_power")
    lane_vec = np.zeros(flow_len)
    o_lanes = int((1 - rho) * NUM_LANES)
    h_lanes = NUM_LANES - o_lanes
    lane_vec[::2] = o_lanes
    lane_vec[1::2] = h_lanes
    flow_per_lane = model.addMVar(flow_len, lb = 0, vtype = GRB.CONTINUOUS, name = "flow_per_lane")
    model.addConstr(flow_per_lane * lane_vec == flow * BPR_A)
    for i in range(flow_len):
        model.addGenConstrPow(flow_per_lane[i], flow_power[i], BPR_POWER)
    latency = model.addMVar(flow_len, lb = 0, vtype = GRB.CONTINUOUS, name = "latency")
    model.addConstr(latency == (flow_power + BPR_B) * DISTANCE)

    ### Add constraints on sigma
    ## sigma_len = len(beta_lst) * segment_type_num * C * S * 2
#    sigma_total_map = np.zeros((len(beta_lst) * segment_type_num, sigma_len))
    sigma_total_row_lst, sigma_total_col_lst, sigma_total_val_lst = [], [], []
    total_vec = np.ones(len(beta_lst) * segment_type_num)
    u_len = len(beta_lst) * segment_type_num * C
#    sigma_conserv_map = np.zeros((u_len, sigma_len))
    sigma_conserv_row_lst, sigma_conserv_col_lst, sigma_conserv_val_lst = [], [], []
    sigma_zero_map = np.zeros(sigma_len)
    zero_vec = np.zeros(sigma_len)
    for grid_idx in range(len(beta_lst)):
        for c in range(C):
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    for s in range(s_o, s_d + 1):
                        sigma_idx = grid_idx * segment_type_num * C * S * 2 + segment_idx * C * S * 2 + c * S * 2 + s * 2
                        total_idx = grid_idx * segment_type_num + segment_idx
#                        sigma_total_map[total_idx, sigma_idx] = 1
#                        sigma_total_map[total_idx, sigma_idx + 1] = 1
                        sigma_total_row_lst += [total_idx, total_idx]
                        sigma_total_col_lst += [sigma_idx, sigma_idx + 1]
                        sigma_total_val_lst += [1, 1]
                        u_idx = grid_idx * segment_type_num * C + segment_idx * C + c
#                        sigma_conserv_map[u_idx, sigma_idx] = 1
#                        sigma_conserv_map[u_idx, sigma_idx + 1] = 1
                        sigma_conserv_row_lst += [u_idx, u_idx]
                        sigma_conserv_col_lst += [sigma_idx, sigma_idx + 1]
                        sigma_conserv_val_lst += [1, 1]
                    sigma_idx_lo = grid_idx * segment_type_num * C * S * 2 + segment_idx * C * S * 2 + c * S * 2
                    sigma_idx_hi = grid_idx * segment_type_num * C * S * 2 + segment_idx * C * S * 2 + c * S * 2 + (s_d + 1) * 2
                    sigma_idx_top = grid_idx * segment_type_num * C * S * 2 + segment_idx * C * S * 2 + c * S * 2 + S * 2
                    sigma_zero_map[sigma_idx:(sigma_idx + s_o * 2)] = 1
                    sigma_zero_map[sigma_idx_hi:sigma_idx_top] = 1
                segment_idx += 1
    sigma_total_map = csr_matrix((sigma_total_val_lst, (sigma_total_row_lst, sigma_total_col_lst)), shape = (len(beta_lst) * segment_type_num, sigma_len))
    sigma_conserv_map = csr_matrix((sigma_conserv_val_lst, (sigma_conserv_row_lst, sigma_conserv_col_lst)), shape = (u_len, sigma_len))
    model.addConstr(sigma_total_map @ sigma == 1)
    u = model.addMVar(u_len, lb = 0, vtype = GRB.CONTINUOUS, name = "u")
    model.addConstr(sigma_conserv_map @ sigma == u)
    model.addConstr(sigma_zero_map * sigma == zero_vec)
    ### Create objective
    objective = ((sigma_to_latency_coef_map @ sigma) * latency).sum() + (sigma_cost_coef * sigma).sum()
    ## Optimize the model
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    obj_val = model.ObjVal
    sigma_ret = np.zeros(sigma_len)
    for i in range(sigma_len):
        sigma_ret[i] = sigma[i].x
    latency_ret = np.zeros(flow_len)
    flow_ret = np.zeros(flow_len)
    for i in range(flow_len):
        latency_ret[i] = latency[i].x
        flow_ret[i] = flow[i].x
    print(latency_ret)
    print(flow_ret)
    return sigma_ret

def solve_equi_sigma(density, hour_idx, rho, tau_cs, num_itr = 10):
    ### Get grid
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    d_len = int(N_HOUR * single_t_d_len * segment_type_num)
    n_grids = len(beta_lst)
    ## Compute sigma to flow map
    sigma_len = n_grids * segment_type_num * C * S * 2
    flow_len = S * 2
    sigma_to_flow_map = np.zeros((flow_len, sigma_len))
    for d_idx in range(single_t_d_len):
        elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
        segment_idx = 0
        for s_o in range(S):
            for s_d in range(s_o, S):
                d_val = density[hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx]
                for s in range(s_o, s_d + 1):
                    for c in range(C):
                        sigma_idx = np.arange(d_idx_start_lst[d_idx] * segment_type_num * C * S * 2 + segment_idx * C * S * 2 + c * S * 2 + s * 2, d_idx_start_lst[d_idx + 1] * segment_type_num * C * S * 2, segment_type_num * C * S * 2)
                        coef_from_d = d_val / elem_num
                        sigma_coef = 1 / (c + 1) * coef_from_d
                        flow_o_idx = 2 * s
                        flow_h_idx = 2 * s + 1
                        sigma_to_flow_map[flow_o_idx, sigma_idx] += sigma_coef
                        sigma_to_flow_map[flow_h_idx, sigma_idx + 1] += sigma_coef
                segment_idx += 1
    ### Compute profile given latencies
    ### n_grid, segment_type_num, C, S
    sigma_target = np.ones(sigma_len)
    loss_arr = []
    for _ in tqdm(range(num_itr)):
        latency = sigma_to_flow_map @ sigma_target
        latency_o, latency_h = latency[::2], latency[1::2]
        sigma_equi = np.zeros(sigma_len)
        sigma_s = solve_sigma_given_parameters_vec(beta_lst, gamma_lst_c, latency_o, latency_h, tau_cs)
        sigma_equi[::2] = 1 - sigma_s[0,:,:,:,:].ravel()
        sigma_equi[1::2] = sigma_s[0,:,:,:,:].ravel()
        sigma_target = (sigma_target + sigma_equi) / 2
        loss = np.mean((sigma_target - sigma_equi) ** 2)
        loss_arr.append(loss)
    print(loss_arr)
    return sigma_target

def describe_sigma(sigma, density, hour_idx):
    beta_lst, gamma_lst_c, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    # len(beta_lst) * segment_type_num * C * S * 2
    segment_type_num = int(S * (S + 1) / 2)
    d_total = density[(hour_idx * single_t_d_len * segment_type_num):((hour_idx + 1) * single_t_d_len * segment_type_num)].sum()
    for segment_idx in range(segment_type_num):
        print(f"Segment {segment_idx}:")
        d_val = density[hour_idx * single_t_d_len * segment_type_num + d_idx * segment_type_num + segment_idx]
        for s in range(S):
            for c in range(C):
                sigma_o_idx = np.arange(segment_idx * C * S * 2 + c * S * 2 + s * 2, len(sigma), segment_type_num * C * S * 2)
                denom = len(beta_lst) * d_val / d_total
                sigma_o_total = sigma[sigma_o_idx].sum() / denom
                sigma_h_total = sigma[sigma_o_idx + 1].sum() / denom
                print(f"\tS = {s}, C = {c + 1}: sigma_o = {sigma_o_total}, sigma_h = {sigma_h_total}")

if DENSITY_RECALIBRATE:
    density = calibrate_density()
    np.save("density/preference_density_general.npy", density)
else:
    density = np.load("density/preference_density_general.npy")
describe_density(density)

#sigma = solve_equi_sigma(density, hour_idx = 12, rho = 0.25, tau_cs = np.array([5, 1.25, 0]).reshape((3, 1)), num_itr = 100)
#describe_sigma(sigma)
