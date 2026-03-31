import json
import math
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy
from scipy import optimize
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, csr_array, dia_matrix, vstack

#from pyomo.environ import ConcreteModel, Var, RangeSet, Constraint, Expression, SolverFactory, value
#from pyomo.mpec import Complementarity, complements

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


###############################################################################
# Script Options
###############################################################################
N_CPU = 30
DENSITY_RECALIBRATE = False
DENSITY_RETRAIN = False
TRAIN_FRAC = 0.8
USE_5_MIN = False
FINE_TUNE = False

###############################################################################
# Hyperparameters
###############################################################################
NUM_LANES = 4
BPR_POWER = 4
BPR_A = 7e-4
BPR_B = 0.7906
DISTANCE = 7.16
WINDOW_SIZE = 1

DELTA = 0.125
num_grids = int(4 / DELTA)

BETA_RANGE_LST_FULL = [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)]
GAMMA_RANGE_DCT_FULL = {
    1: [(0, 0)],
    2: [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)],
    3: [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)]
}

BETA_RANGE_LST_AM = [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2)]
GAMMA_RANGE_DCT_AM = {
    1: [(0, 0)],
    2: [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2)],
    3: [(0, 0.25), (0.25, 0.5), (0.5, 1)]
}

BETA_RANGE_LST = BETA_RANGE_LST_FULL
GAMMA_RANGE_DCT = GAMMA_RANGE_DCT_FULL

C = 3
INT_GRID = 1


###############################################################################
# Load Data
###############################################################################
if not USE_5_MIN:
    GROUPBY_COLS = ["Date", "Hour"]
    df = pd.read_csv("data/df_meta.csv")
else:
    GROUPBY_COLS = ["Date", "Hour", "Minute"]
    df = pd.read_csv("data/df_meta_5min.csv")

df_pop = pd.read_csv("pop_fraction.csv", thousands=",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df = df.sort_values(GROUPBY_COLS, ascending=True)

data_cols = ['HOV Travel Time', 'Ordinary Travel Time', 'Avg_total_toll']
for col in data_cols:
    df[col] = df.groupby(["Segment"])[col].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=False).mean()
    )

df = df[(df["Date"] >= "2021-02-01") & (df["Date"] <= "2021-05-31")]
df = df[(df["Hour"] >= 7) & (df["Hour"] <= 18)]
df = df.dropna()

df_wide = df.pivot(
    index=GROUPBY_COLS,
    columns=["Segment"],
    values=["HOV Flow", "Ordinary Flow", "HOV Travel Time", "Ordinary Travel Time", "Avg_total_toll"]
)
df_wide.columns = [x + "_" + y for x, y in df_wide.columns]
segment_lst = [x.split("_")[1].strip() for x in df_wide.columns if "HOV Flow" in x]
S = len(segment_lst)

DISTANCE_ARR = np.zeros(S)
for segment_idx in range(len(segment_lst)):
    distance = df[df["Segment"] == segment_lst[segment_idx]].iloc[0]["Distance"]
    DISTANCE_ARR[segment_idx] = distance

df_wide = df_wide.dropna().reset_index()

df = df[df["Ordinary Travel Time"] > df["HOV Travel Time"]]
df = df.sort_values(GROUPBY_COLS, ascending=True)

df_pop["Sigma_1ratio"] = df_pop["Single"] / (
    df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3
)
df_pop["Sigma_2ratio"] = df_pop["TwoPeople"] * 2 / (
    df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3
)
df_pop["Sigma_3ratio"] = df_pop["ThreePlus"] * 3 / (
    df_pop["Single"] + df_pop["TwoPeople"] * 2 + df_pop["ThreePlus"] * 3
)
df = df.merge(df_pop[["Date", "Sigma_1ratio", "Sigma_2ratio", "Sigma_3ratio"]], on="Date")
df = df.sort_values(GROUPBY_COLS, ascending=True)
df = df.dropna()

TAU_LST = np.array(df["Avg_total_toll"])
N_DATA = df_wide.shape[0]

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

for segment_idx in range(len(segment_lst)):
    segment = segment_lst[segment_idx]
    TAU_CS_LST[:, 0, segment_idx] = np.array(df_wide[f"Avg_total_toll_{segment}"])
    TAU_CS_LST[:, 1, segment_idx] = TAU_CS_LST[:, 0, segment_idx] / 4
    LATENCY_O_LST[:, segment_idx] = np.array(df_wide[f"Ordinary Travel Time_{segment}"])
    LATENCY_HOV_LST[:, segment_idx] = np.array(df_wide[f"HOV Travel Time_{segment}"])
    FLOW_O_LST[(N_DATA * segment_idx):(N_DATA * (segment_idx + 1))] = np.array(df_wide[f"Ordinary Flow_{segment}"])
    FLOW_HOV_LST[(N_DATA * segment_idx):(N_DATA * (segment_idx + 1))] = np.array(df_wide[f"HOV Flow_{segment}"])
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
            HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] = df_od_demand[
                (df_od_demand["Hour"] == hour)
                & (df_od_demand["Origin"] == origin_seg)
                & (df_od_demand["Destination"] == dest_seg)
            ].iloc[0]["Demand"]
            if USE_5_MIN:
                HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] /= 12
            segment_idx += 1

RATIO_INDEX_TO_IGNORE = [22, 39, 40, 86]
DATES_TO_IGNORE = ["2021-02-15", "2021-03-31", "2021-04-23", "2021-04-26", "2021-04-27", "2021-04-28", "2021-06-30"]

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


###############################################################################
# Utilities
###############################################################################
class STEArgmin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        index = torch.argmin(input, dim=-1)
        ctx.save_for_backward(input, index)
        return index

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        softmin = torch.softmax(-input, dim=-1)
        dot = (grad_output * softmin).sum(dim=-1, keepdim=True)
        grad_input = softmin * (grad_output - dot)
        return grad_input


def ste_argmin(input):
    return STEArgmin.apply(input)

class STEOneHotArgmin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        input: (..., K)
        returns: hard one-hot tensor of shape (..., K),
                 selecting argmin along the last dimension
        """
        index = torch.argmin(input, dim=-1)  # (...,)
        hard = torch.nn.functional.one_hot(index, num_classes=input.shape[-1]).to(input.dtype)
        ctx.save_for_backward(input)
        return hard

    @staticmethod
    def backward(ctx, grad_output, beta = 100.0):
        """
        Use the Jacobian of softmin in the backward pass.
        grad_output: (..., K)
        returns: grad_input of shape (..., K)
        """
        (input,) = ctx.saved_tensors
        softmin = torch.softmax(-input * beta, dim=-1)  # (..., K)

        # Jacobian-vector product for softmin:
        # d softmin(x) = -[Diag(s) - s s^T] dx
        # Equivalent VJP with incoming grad_output:
        dot = (grad_output * softmin).sum(dim=-1, keepdim=True)
        grad_input = softmin * (dot - grad_output)
        return grad_input

def ste_onehot_argmin(input):
    return STEOneHotArgmin.apply(input)


def get_cost(flow, distance, bpr_a=BPR_A, bpr_b=BPR_B):
    return ((bpr_a * flow) ** BPR_POWER + bpr_b) * distance


###############################################################################
# Best response for a single exact type (kept for compatibility / debugging)
###############################################################################
def solve_sigma_given_parameters(beta, gamma_c, c_o, c_h, tau_cs):
    C_loc, S_loc = tau_cs.shape
    cost_o = beta * c_o
    cost_h = beta * c_h + gamma_c.reshape((C_loc, 1)) + tau_cs
    lane_cs = (cost_h < cost_o) + 0
    total_cost_c = np.sum(lane_cs * cost_h + (1 - lane_cs) * cost_o, axis=1)
    best_c = np.argmin(total_cost_c)
    return lane_cs[best_c, :]


###############################################################################
# Coarse density cells and fine beta grid
###############################################################################
def normalize_cell(cell):
    """
    Convert a coarse cell description into a hashable tuple.
    cell is like:
        [beta_interval, gamma1_interval, gamma2_interval, gamma3_interval]
    """
    return tuple(tuple(x) for x in cell)

def get_hourly_density_feasibility_mask(
    unique_hour_lst,
    beta_range_lst_full=BETA_RANGE_LST_FULL,
    gamma_range_dct_full=GAMMA_RANGE_DCT_FULL,
    beta_range_lst_am=BETA_RANGE_LST_AM,
    gamma_range_dct_am=GAMMA_RANGE_DCT_AM,
):
    """
    Returns
    -------
    allowed_mask : np.ndarray, shape (N_HOUR, single_t_d_len), dtype=bool
        allowed_mask[h, d] = True if coarse density cell d is allowed at hour index h.
    """
    full_cells = get_beta_gamma_range_lst(
        beta_range_lst=beta_range_lst_full,
        gamma_range_dct=gamma_range_dct_full
    )
    am_cells = get_beta_gamma_range_lst(
        beta_range_lst=beta_range_lst_am,
        gamma_range_dct=gamma_range_dct_am
    )

    full_cells_norm = [normalize_cell(cell) for cell in full_cells]
    am_cell_set = set(normalize_cell(cell) for cell in am_cells)

    single_t_d_len = len(full_cells_norm)
    allowed_mask = np.zeros((len(unique_hour_lst), single_t_d_len), dtype=bool)

    for hour_idx, hour in enumerate(unique_hour_lst):
        if 7 <= hour <= 12:
            for d_idx, cell in enumerate(full_cells_norm):
                allowed_mask[hour_idx, d_idx] = (cell in am_cell_set)
        elif 13 <= hour <= 18:
            allowed_mask[hour_idx, :] = True
        else:
            # default: allow full domain unless you want another rule
            allowed_mask[hour_idx, :] = True

    return allowed_mask
    
def get_beta_gamma_range_lst(beta_range_lst=BETA_RANGE_LST, gamma_range_dct=GAMMA_RANGE_DCT):
    beta_gamma_range_lst = [[x] for x in beta_range_lst]
    for c in range(C):
        tmp = []
        for lst in beta_gamma_range_lst:
            for tup in gamma_range_dct[c + 1]:
                elem = lst.copy() + [tup]
                tmp.append(elem)
        beta_gamma_range_lst = tmp
    return beta_gamma_range_lst


def get_grid(beta_range_lst=BETA_RANGE_LST, gamma_range_dct=GAMMA_RANGE_DCT):
    """
    Coarse density cells remain over (beta interval, gamma intervals).
    Inside each coarse cell, only beta is refined.

    Returns
    -------
    beta_lst : (n_grids,)
    gamma_box_map : (n_grids, C-1, 2)
        For each fine beta point, store the gamma increment box:
            gamma_box_map[g, k, 0] = lower bound of increment for occupancy k+2
            gamma_box_map[g, k, 1] = upper bound
    d_idx_start_lst : (n_cells + 1,)
    """
    eps = 1e-9
    beta_gamma_range_lst = get_beta_gamma_range_lst(
        beta_range_lst=beta_range_lst,
        gamma_range_dct=gamma_range_dct
    )

    beta_lst_all = []
    gamma_box_all = []
    d_idx_start_lst = [0]

    for lst in beta_gamma_range_lst:
        beta_lo, beta_hi = lst[0]
        beta_curr = np.arange(beta_lo, beta_hi + eps, DELTA)

        gamma_box = []
        for occ in range(2, C + 1):
            gamma_box.append(lst[occ])
        gamma_box = np.array(gamma_box, dtype=float)

        beta_lst_all.append(beta_curr)
        gamma_box_all.append(np.tile(gamma_box[None, :, :], (len(beta_curr), 1, 1)))
        d_idx_start_lst.append(d_idx_start_lst[-1] + len(beta_curr))

    beta_lst = np.concatenate(beta_lst_all)
    gamma_box_map = np.concatenate(gamma_box_all, axis=0)
    d_idx_start_lst = np.array(d_idx_start_lst, dtype=int)

    return beta_lst, gamma_box_map, d_idx_start_lst


###############################################################################
# Occupancy fractions inside a coarse gamma box
###############################################################################
def _interval_len_np(lo, hi):
    return max(hi - lo, 0.0)


def _interval_len_torch(lo, hi):
    return torch.clamp(hi - lo, min=0.0)

def occupancy_fraction_from_gamma_box_batch(tilde_cost_c, gamma_box_map):
    """
    Vectorized occupancy fractions over all grids and OD pairs.

    Parameters
    ----------
    tilde_cost_c : np.ndarray, shape (G, M, C)
        tilde_cost_c[g, m, c] = \tilde C_c^{ij}(beta_g) for OD pair m.
    gamma_box_map : np.ndarray, shape (G, C-1, 2)
        For each fine beta grid point g, the coarse gamma increment box.

    Returns
    -------
    share : np.ndarray, shape (G, M, C)
        Fraction of the coarse gamma box choosing each occupancy.
    """
    G, M, C_loc = tilde_cost_c.shape
    dtype = np.float32

    if C_loc == 1:
        return np.ones((G, M, 1), dtype=dtype)

    if C_loc == 2:
        l2 = gamma_box_map[:, 0, 0][:, None]   # (G,1)
        u2 = gamma_box_map[:, 0, 1][:, None]   # (G,1)

        K1 = tilde_cost_c[:, :, 0]
        K2 = tilde_cost_c[:, :, 1]

        total_len = np.clip(u2 - l2, 0.0, None)
        thresh = K1 - K2
        len_occ1 = np.clip(u2 - np.maximum(l2, thresh), 0.0, None)

        frac1 = np.where(total_len > 1e-12, len_occ1 / total_len, (K1 <= l2 + K2).astype(dtype))
        frac2 = 1.0 - frac1
        return np.stack([frac1, frac2], axis=-1).astype(dtype)

    if C_loc != 3:
        raise NotImplementedError("This batched implementation currently supports only C <= 3.")

    # C = 3
    l2 = gamma_box_map[:, 0, 0][:, None]   # (G,1)
    u2 = gamma_box_map[:, 0, 1][:, None]
    l3 = gamma_box_map[:, 1, 0][:, None]
    u3 = gamma_box_map[:, 1, 1][:, None]

    K1 = tilde_cost_c[:, :, 0]   # (G,M)
    K2 = tilde_cost_c[:, :, 1]
    K3 = tilde_cost_c[:, :, 2]

    W = np.clip(u2 - l2, 0.0, None)
    H = np.clip(u3 - l3, 0.0, None)
    total_area = W * H

    a = K1 - K2
    b = K1 - K3
    c = K2 - K3

    # Occupancy 2:
    # x <= a, y >= c
    area2 = (
        np.clip(np.minimum(u2, a) - l2, 0.0, None)
        * np.clip(u3 - np.maximum(l3, c), 0.0, None)
    )

    # Occupancy 3:
    # x + y <= b, y <= c
    y1 = np.minimum(u3, c)

    flat_hi = np.minimum(y1, b - u2)
    flat = W * np.clip(flat_hi - l3, 0.0, None)

    lin_lo = np.maximum(l3, b - u2)
    lin_hi = np.minimum(y1, b - l2)
    lin_len = np.clip(lin_hi - lin_lo, 0.0, None)
    area3 = flat + np.where(
        lin_len > 0,
        (b - l2) * lin_len - 0.5 * (lin_hi ** 2 - lin_lo ** 2),
        0.0
    )

    area1 = np.clip(total_area - area2 - area3, 0.0, None)

    share = np.stack([area1, area2, area3], axis=-1).astype(dtype)   # (G,M,3)
    s = share.sum(axis=-1, keepdims=True)
    share = share / s

#    fallback_costs = np.stack(
#        [
#            K1,
#            l2 + K2,
#            l2 + l3 + K3,
#        ],
#        axis=-1
#    )  # (G,M,3)
#    best = np.argmin(fallback_costs, axis=-1)   # (G,M)
#    fallback = np.eye(3, dtype=dtype)[best]     # (G,M,3)
#
#    share = np.where(s > 1e-12, share / np.maximum(s, 1e-12), fallback)
    return share.astype(dtype)

def occupancy_fraction_from_gamma_box_torch_batch(tilde_cost_c, gamma_box_map):
    """
    Vectorized occupancy fractions over all grids and OD pairs.

    Parameters
    ----------
    tilde_cost_c : torch.Tensor, shape (G, M, C)
        tilde_cost_c[g, m, c] = \tilde C_c^{ij}(beta_g) for OD pair m.
    gamma_box_map : torch.Tensor, shape (G, C-1, 2)
        For each fine beta grid point g, the coarse gamma increment box.
        gamma_box_map[g, k, 0] = lower bound
        gamma_box_map[g, k, 1] = upper bound
        for increment corresponding to occupancy k+2.

    Returns
    -------
    share : torch.Tensor, shape (G, M, C)
        Fraction of the coarse gamma box choosing each occupancy.
    """
    G, M, C_loc = tilde_cost_c.shape
    dtype = tilde_cost_c.dtype
    device = tilde_cost_c.device

    if C_loc == 1:
        return torch.ones((G, M, 1), dtype=dtype, device=device)

    if C_loc == 2:
        l2 = gamma_box_map[:, 0, 0].unsqueeze(1)   # (G,1)
        u2 = gamma_box_map[:, 0, 1].unsqueeze(1)   # (G,1)

        K1 = tilde_cost_c[:, :, 0]
        K2 = tilde_cost_c[:, :, 1]

        total_len = torch.clamp(u2 - l2, min=0.0)
        thresh = K1 - K2
        len_occ1 = torch.clamp(u2 - torch.maximum(l2, thresh), min=0.0)

        frac1 = torch.where(
            total_len > 1e-12,
            len_occ1 / total_len,
            (K1 <= l2 + K2).to(dtype)
        )
        frac2 = 1.0 - frac1
        return torch.stack([frac1, frac2], dim=-1)

    if C_loc != 3:
        raise NotImplementedError("This batched implementation currently supports only C <= 3.")

    # C = 3
    # x = gamma_2 increment, y = gamma_3 increment
    # cumulative disutility:
    #   G1 = 0
    #   G2 = x
    #   G3 = x + y

    l2 = gamma_box_map[:, 0, 0].unsqueeze(1)   # (G,1)
    u2 = gamma_box_map[:, 0, 1].unsqueeze(1)
    l3 = gamma_box_map[:, 1, 0].unsqueeze(1)
    u3 = gamma_box_map[:, 1, 1].unsqueeze(1)

    K1 = tilde_cost_c[:, :, 0]  # (G,M)
    K2 = tilde_cost_c[:, :, 1]
    K3 = tilde_cost_c[:, :, 2]

    W = torch.clamp(u2 - l2, min=0.0)
    H = torch.clamp(u3 - l3, min=0.0)
    total_area = W * H

    a = K1 - K2
    b = K1 - K3
    c = K2 - K3

    # Occupancy 2:
    # x + K2 <= K1      => x <= K1 - K2 = a
    # x + K2 <= x+y+K3  => y >= K2 - K3 = c
    area2 = (
        torch.clamp(torch.minimum(u2, a) - l2, min=0.0)
        * torch.clamp(u3 - torch.maximum(l3, c), min=0.0)
    )

    # Occupancy 3:
    # x+y+K3 <= K1      => x+y <= K1 - K3 = b
    # x+y+K3 <= x+K2    => y <= K2 - K3 = c
    y1 = torch.minimum(u3, c)

    # Flat part where width in x is full W
    flat_hi = torch.minimum(y1, b - u2)
    flat = W * torch.clamp(flat_hi - l3, min=0.0)

    # Sloped part where width = b - y - l2
    lin_lo = torch.maximum(l3, b - u2)
    lin_hi = torch.minimum(y1, b - l2)
    lin_len = torch.clamp(lin_hi - lin_lo, min=0.0)
    area3 = flat + torch.where(
        lin_len > 0,
        (b - l2) * lin_len - 0.5 * (lin_hi ** 2 - lin_lo ** 2),
        torch.zeros_like(lin_len)
    )

    area1 = torch.clamp(total_area - area2 - area3, min=0.0)

    share = torch.stack([area1, area2, area3], dim=-1)  # (G,M,3)
    s = share.sum(dim=-1, keepdim=True)
    share = share / s

#    # Numerical fallback at lower-left corner of the box
#    fallback_costs = torch.stack(
#        [
#            K1,
#            l2 + K2,
#            l2 + l3 + K3,
#        ],
#        dim=-1
#    )  # (G,M,3)
#    best = torch.argmin(fallback_costs, dim=-1)  # (G,M)
#    fallback = F.one_hot(best, num_classes=3).to(dtype)
#
#    share = torch.where(s > 1e-12, share / s, fallback)
    return share


def gamma_midpoint_map_from_boxes(gamma_box_map):
    """
    Used only for utility-cost bookkeeping.
    Convert increment-box midpoints to cumulative gamma midpoints.
    """
    n = gamma_box_map.shape[0]
    out = np.zeros((n, C), dtype=float)
    if C == 1:
        return out
    inc_mid = gamma_box_map.mean(axis=2)  # (n, C-1)
    out[:, 1:] = np.cumsum(inc_mid, axis=1)
    return out


###############################################################################
# Vectorized best responses
###############################################################################
def solve_sigma_given_parameters_vec(beta_lst, gamma_box_map, c_o, c_h, tau_cs):
    """
    Batched NumPy implementation.

    Parameters
    ----------
    beta_lst : np.ndarray, shape (G,)
    gamma_box_map : np.ndarray, shape (G, C-1, 2)
    c_o : np.ndarray, shape (S,)
    c_h : np.ndarray, shape (S,)
    tau_cs : np.ndarray, shape (C, S)

    Returns
    -------
    lane_cs_h : np.ndarray, shape (1, G, M, C, S)
    lane_cs_o : np.ndarray, shape (1, G, M, C, S)
    occ_frac  : np.ndarray, shape (1, G, M, C)
    """
    assert beta_lst.shape[0] == gamma_box_map.shape[0]

    C_loc, S_loc = tau_cs.shape
    G = beta_lst.shape[0]
    M = int(S_loc * (S_loc + 1) / 2)
    dtype = np.float32

    # ------------------------------------------------------------
    # 1) Lane choice for each occupancy and segment, batched
    # ------------------------------------------------------------
    cost_o = beta_lst[:, None] * c_o[None, :]                              # (G,S)
    cost_h = beta_lst[:, None, None] * c_h[None, None, :] + tau_cs[None, :, :]  # (G,C,S)

    lane_cs = (cost_h < cost_o[:, None, :]).astype(dtype)                  # (G,C,S)
    total_cost_mat = lane_cs * cost_h + (1.0 - lane_cs) * cost_o[:, None, :]  # (G,C,S)

    # ------------------------------------------------------------
    # 2) Build all OD pairs and active masks once
    # ------------------------------------------------------------
    od_starts = []
    od_ends = []
    active_masks = []

    for s_o in range(S_loc):
        for s_d in range(s_o, S_loc):
            od_starts.append(s_o)
            od_ends.append(s_d)

            mask = np.zeros(S_loc, dtype=dtype)
            mask[s_o:(s_d + 1)] = 1.0
            active_masks.append(mask)

    od_starts = np.array(od_starts, dtype=np.int64)   # (M,)
    od_ends = np.array(od_ends, dtype=np.int64)       # (M,)
    active_masks = np.stack(active_masks, axis=0)     # (M,S)

    # ------------------------------------------------------------
    # 3) Compute tilde C_c^{ij}(beta) for all grids and OD pairs
    #    using prefix sums
    # ------------------------------------------------------------
    prefix = np.cumsum(total_cost_mat, axis=2)                                # (G,C,S)
    prefix_pad = np.concatenate(
        [np.zeros((G, C_loc, 1), dtype=dtype), prefix.astype(dtype)],
        axis=2
    )                                                                         # (G,C,S+1)

    # Advanced indexing gives arrays of shape (G,C,M)
    ends_val = prefix_pad[:, :, od_ends + 1]
    starts_val = prefix_pad[:, :, od_starts]
    tilde_cost_pairs = ends_val - starts_val                                  # (G,C,M)
    tilde_cost_pairs = np.transpose(tilde_cost_pairs, (0, 2, 1))             # (G,M,C)

    # ------------------------------------------------------------
    # 4) Occupancy fractions for all grids and OD pairs
    # ------------------------------------------------------------
    occ_frac = occupancy_fraction_from_gamma_box_batch(
        tilde_cost_pairs, gamma_box_map
    ).astype(dtype)                                                           # (G,M,C)

    # ------------------------------------------------------------
    # 5) Build final expected lane-choice tensors by broadcasting
    # ------------------------------------------------------------
    lane_cs_exp = lane_cs[:, None, :, :]                   # (G,1,C,S)
    occ_frac_exp = occ_frac[:, :, :, None]                # (G,M,C,1)
    active_masks_exp = active_masks[None, :, None, :]     # (1,M,1,S)

    lane_cs_h = occ_frac_exp * lane_cs_exp * active_masks_exp
    lane_cs_o = occ_frac_exp * (1.0 - lane_cs_exp) * active_masks_exp

    # Add leading n_data dimension = 1 for compatibility
    lane_cs_h = lane_cs_h[None, :, :, :, :]   # (1,G,M,C,S)
    lane_cs_o = lane_cs_o[None, :, :, :, :]   # (1,G,M,C,S)
    occ_frac = occ_frac[None, :, :, :]        # (1,G,M,C)

    return lane_cs_h.astype(dtype), lane_cs_o.astype(dtype), occ_frac.astype(dtype)

def solve_sigma_given_parameters_vec_torch_hardmin(beta_lst, gamma_box_map, c_o, c_h, tau_cs):
    """
    Batched torch implementation.

    Parameters
    ----------
    beta_lst : torch.Tensor, shape (G,)
    gamma_box_map : torch.Tensor, shape (G, C-1, 2)
    c_o : torch.Tensor, shape (S,)
    c_h : torch.Tensor, shape (S,)
    tau_cs : torch.Tensor, shape (C, S)

    Returns
    -------
    lane_cs_h : torch.Tensor, shape (1, G, M, C, S)
    lane_cs_o : torch.Tensor, shape (1, G, M, C, S)
    occ_frac  : torch.Tensor, shape (1, G, M, C)
    """
    assert beta_lst.shape[0] == gamma_box_map.shape[0]

    dtype = beta_lst.dtype
    device = beta_lst.device

    C_loc, S_loc = tau_cs.shape
    G = beta_lst.shape[0]
    M = int(S_loc * (S_loc + 1) / 2)

    # ------------------------------------------------------------------
    # 1) Lane choice for each occupancy and segment, batched over grids
    # ------------------------------------------------------------------
    # cost_o: (G, S)
    cost_o = beta_lst[:, None] * c_o[None, :]

    # cost_h: (G, C, S)
    cost_h = beta_lst[:, None, None] * c_h[None, None, :] + tau_cs[None, :, :]

    # lane_cs[g,c,s] = 1 if HOT is chosen for occupancy c on segment s
    lane_cs = (cost_h < cost_o[:, None, :]).to(dtype)  # (G, C, S)

    # minimized lane cost for each occupancy and segment
    total_cost_mat = lane_cs * cost_h + (1.0 - lane_cs) * cost_o[:, None, :]  # (G,C,S)

    # ------------------------------------------------------------------
    # 2) Build all OD pairs and active masks once
    # ------------------------------------------------------------------
    od_starts = []
    od_ends = []
    active_masks = []

    for s_o in range(S_loc):
        for s_d in range(s_o, S_loc):
            od_starts.append(s_o)
            od_ends.append(s_d)

            mask = torch.zeros(S_loc, dtype=dtype, device=device)
            mask[s_o:(s_d + 1)] = 1.0
            active_masks.append(mask)

    od_starts = torch.tensor(od_starts, dtype=torch.long, device=device)  # (M,)
    od_ends = torch.tensor(od_ends, dtype=torch.long, device=device)      # (M,)
    active_masks = torch.stack(active_masks, dim=0)                       # (M,S)

    # ------------------------------------------------------------------
    # 3) Compute \tilde C_c^{ij}(beta) for all grids and all OD pairs
    #    using prefix sums
    # ------------------------------------------------------------------
    # prefix_pad: (G, C, S+1), with prefix_pad[:,:,0] = 0
    prefix = torch.cumsum(total_cost_mat, dim=2)
    prefix_pad = torch.cat(
        [torch.zeros((G, C_loc, 1), dtype=dtype, device=device), prefix],
        dim=2
    )

    # Gather prefix sums at OD endpoints
    # ends_idx:   (G, C, M)
    # starts_idx: (G, C, M)
    ends_idx = (od_ends + 1).view(1, 1, M).expand(G, C_loc, M)
    starts_idx = od_starts.view(1, 1, M).expand(G, C_loc, M)

    tilde_cost_pairs = prefix_pad.gather(2, ends_idx) - prefix_pad.gather(2, starts_idx)  # (G,C,M)
    tilde_cost_pairs = tilde_cost_pairs.permute(0, 2, 1).contiguous()  # (G,M,C)

    # ------------------------------------------------------------------
    # 4) Occupancy fractions for all grids and all OD pairs
    # ------------------------------------------------------------------
    occ_frac = occupancy_fraction_from_gamma_box_torch_batch(
        tilde_cost_pairs, gamma_box_map
    )  # (G,M,C)

    # ------------------------------------------------------------------
    # 5) Build final expected lane-choice tensors by broadcasting
    # ------------------------------------------------------------------
    # lane_cs:      (G, C, S)     -> (G, 1, C, S)
    # occ_frac:     (G, M, C)     -> (G, M, C, 1)
    # active_masks: (M, S)        -> (1, M, 1, S)
    lane_cs_exp = lane_cs.unsqueeze(1)                 # (G,1,C,S)
    occ_frac_exp = occ_frac.unsqueeze(-1)              # (G,M,C,1)
    active_masks_exp = active_masks.unsqueeze(0).unsqueeze(2)  # (1,M,1,S)

    lane_cs_h = occ_frac_exp * lane_cs_exp * active_masks_exp
    lane_cs_o = occ_frac_exp * (1.0 - lane_cs_exp) * active_masks_exp

    # Add leading n_data dimension = 1 for compatibility
    lane_cs_h = lane_cs_h.unsqueeze(0)  # (1,G,M,C,S)
    lane_cs_o = lane_cs_o.unsqueeze(0)  # (1,G,M,C,S)
    occ_frac = occ_frac.unsqueeze(0)    # (1,G,M,C)

    return lane_cs_h, lane_cs_o, occ_frac

def solve_sigma_given_parameters_vec_torch(beta_lst, gamma_box_map, c_o, c_h, tau_cs):
    """
    Batched torch implementation.

    Parameters
    ----------
    beta_lst : torch.Tensor, shape (G,)
    gamma_box_map : torch.Tensor, shape (G, C-1, 2)
    c_o : torch.Tensor, shape (S,)
    c_h : torch.Tensor, shape (S,)
    tau_cs : torch.Tensor, shape (C, S)

    Returns
    -------
    lane_cs_h : torch.Tensor, shape (1, G, M, C, S)
    lane_cs_o : torch.Tensor, shape (1, G, M, C, S)
    occ_frac  : torch.Tensor, shape (1, G, M, C)
    """
    assert beta_lst.shape[0] == gamma_box_map.shape[0]

    dtype = beta_lst.dtype
    device = beta_lst.device

    C_loc, S_loc = tau_cs.shape
    G = beta_lst.shape[0]
    M = int(S_loc * (S_loc + 1) / 2)

    # ------------------------------------------------------------------
    # 1) Lane choice for each occupancy and segment, batched over grids
    # ------------------------------------------------------------------
    # cost_o: (G, S)
    cost_o = beta_lst[:, None] * c_o[None, :]

    # cost_h: (G, C, S)
    cost_h = beta_lst[:, None, None] * c_h[None, None, :] + tau_cs[None, :, :]

    # Build two-way costs for each (g, c, s):
    # choice_cost[..., 0] = HOT cost
    # choice_cost[..., 1] = ordinary-lane cost
    choice_cost = torch.stack(
        [cost_h, cost_o[:, None, :].expand(-1, C_loc, -1)],
        dim=-1,
    )  # (G, C, S, 2)

    # Hard forward argmin, softmin gradient backward
    lane_onehot = ste_onehot_argmin(choice_cost)  # (G, C, S, 2)

    # lane_cs[g,c,s] = 1 if HOT is chosen, 0 otherwise
    lane_cs = lane_onehot[..., 0]  # (G, C, S)

    # minimized lane cost for each occupancy and segment
    total_cost_mat = (
        lane_onehot[..., 0] * cost_h
        + lane_onehot[..., 1] * cost_o[:, None, :]
    )  # (G, C, S)

    # ------------------------------------------------------------------
    # 2) Build all OD pairs and active masks once
    # ------------------------------------------------------------------
    od_starts = []
    od_ends = []
    active_masks = []

    for s_o in range(S_loc):
        for s_d in range(s_o, S_loc):
            od_starts.append(s_o)
            od_ends.append(s_d)

            mask = torch.zeros(S_loc, dtype=dtype, device=device)
            mask[s_o:(s_d + 1)] = 1.0
            active_masks.append(mask)

    od_starts = torch.tensor(od_starts, dtype=torch.long, device=device)  # (M,)
    od_ends = torch.tensor(od_ends, dtype=torch.long, device=device)      # (M,)
    active_masks = torch.stack(active_masks, dim=0)                       # (M,S)

    # ------------------------------------------------------------------
    # 3) Compute \tilde C_c^{ij}(beta) for all grids and all OD pairs
    #    using prefix sums
    # ------------------------------------------------------------------
    prefix = torch.cumsum(total_cost_mat, dim=2)
    prefix_pad = torch.cat(
        [torch.zeros((G, C_loc, 1), dtype=dtype, device=device), prefix],
        dim=2
    )

    ends_idx = (od_ends + 1).view(1, 1, M).expand(G, C_loc, M)
    starts_idx = od_starts.view(1, 1, M).expand(G, C_loc, M)

    tilde_cost_pairs = prefix_pad.gather(2, ends_idx) - prefix_pad.gather(2, starts_idx)  # (G,C,M)
    tilde_cost_pairs = tilde_cost_pairs.permute(0, 2, 1).contiguous()  # (G,M,C)

    # ------------------------------------------------------------------
    # 4) Occupancy fractions for all grids and all OD pairs
    # ------------------------------------------------------------------
    occ_frac = occupancy_fraction_from_gamma_box_torch_batch(
        tilde_cost_pairs, gamma_box_map
    )  # (G,M,C)

    # ------------------------------------------------------------------
    # 5) Build final expected lane-choice tensors by broadcasting
    # ------------------------------------------------------------------
    lane_cs_exp = lane_cs.unsqueeze(1)                 # (G,1,C,S)
    occ_frac_exp = occ_frac.unsqueeze(-1)              # (G,M,C,1)
    active_masks_exp = active_masks.unsqueeze(0).unsqueeze(2)  # (1,M,1,S)

    lane_cs_h = occ_frac_exp * lane_cs_exp * active_masks_exp
    lane_cs_o = occ_frac_exp * (1.0 - lane_cs_exp) * active_masks_exp

    # Add leading n_data dimension = 1 for compatibility
    lane_cs_h = lane_cs_h.unsqueeze(0)  # (1,G,M,C,S)
    lane_cs_o = lane_cs_o.unsqueeze(0)  # (1,G,M,C,S)
    occ_frac = occ_frac.unsqueeze(0)    # (1,G,M,C)

    return lane_cs_h, lane_cs_o, occ_frac


###############################################################################
# Profile generation
###############################################################################
def profile_given_data_single(
    lo,
    hi,
    beta_lst,
    gamma_box_map,
    segment_type_num,
    latency_o_lst=LATENCY_O_LST,
    latency_hov_lst=LATENCY_HOV_LST,
    tau_cs_lst=TAU_CS_LST
):
    N_DATA_loc, C_loc, S_loc = tau_cs_lst.shape
    sigma_ns_h = np.zeros((N_DATA_loc, len(beta_lst), segment_type_num, C_loc, S_loc), dtype=np.float32)
    sigma_ns_o = np.zeros((N_DATA_loc, len(beta_lst), segment_type_num, C_loc, S_loc), dtype=np.float32)
    occ_frac_ns = np.zeros((N_DATA_loc, len(beta_lst), segment_type_num, C_loc), dtype=np.float32)

    for data_idx in tqdm(range(lo, hi)):
        sigma_s_h, sigma_s_o, occ_frac = solve_sigma_given_parameters_vec(
            beta_lst,
            gamma_box_map,
            latency_o_lst[data_idx, :],
            latency_hov_lst[data_idx, :],
            tau_cs_lst[data_idx, :, :]
        )
        sigma_ns_h[data_idx, :, :, :, :] = sigma_s_h[0, :, :, :, :]
        sigma_ns_o[data_idx, :, :, :, :] = sigma_s_o[0, :, :, :, :]
        occ_frac_ns[data_idx, :, :, :] = occ_frac[0, :, :, :]

    return sigma_ns_h, sigma_ns_o, occ_frac_ns


###############################################################################
# Identifiability helpers
###############################################################################
def get_d_coef_matrix(sigma_ns_h, sigma_ns_o, meta_data=None, data_dct=None):
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

    beta_lst, gamma_box_map, d_idx_start_lst = get_grid(
        beta_range_lst=BETA_RANGE_LST,
        gamma_range_dct=GAMMA_RANGE_DCT
    )
    segment_type_num = int(S * (S + 1) / 2)
    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len)
    
    HOURLY_DENSITY_ALLOWED_MASK = get_hourly_density_feasibility_mask(
        UNIQUE_HOUR_LST,
        beta_range_lst_full=BETA_RANGE_LST_FULL,
        gamma_range_dct_full=GAMMA_RANGE_DCT_FULL,
        beta_range_lst_am=BETA_RANGE_LST_AM,
        gamma_range_dct_am=GAMMA_RANGE_DCT_AM,
    )
    
    dropped_cols = []
    for hour_idx in range(N_HOUR):
        for k in range(single_t_d_len):
            if not HOURLY_DENSITY_ALLOWED_MASK[hour_idx, k]:
                dropped_cols.append(hour_idx * single_t_d_len + k)

    d_coef_matrix = np.zeros((2 * N_DATA + len(dropped_cols) + 1, d_len))
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
                            d_coef_matrix[relev_data_idx, hour_idx * single_t_d_len + d_idx] += (
                                1 / (c + 1)
                                * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], segment_idx, c, s].sum(axis=1)
                                / elem_num
                                * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            )
                            d_coef_matrix[N_DATA + relev_data_idx, hour_idx * single_t_d_len + d_idx] += (
                                1 / (c + 1)
                                * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], segment_idx, c, s].sum(axis=1)
                                / elem_num
                                * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            )
                    segment_idx += 1
    for i,col in enumerate(dropped_cols):
        d_coef_matrix[2 * N_DATA + i, col] = 1
    d_coef_matrix[-1, :] = 1
    return d_coef_matrix


def drop_dependent_columns(X, tol=1e-10):
    Q, R, pivot = scipy.linalg.qr(X, mode='economic', pivoting=True)
    diag_R = np.abs(np.diag(R))
    rank = np.sum(diag_R > tol)
    idx_indep = sorted(pivot[:rank])
    idx_dropped = sorted(pivot[rank:])
    X_indep = X[:, idx_indep]
    return X_indep, idx_dropped


def is_identifiable(sigma_ns_h, sigma_ns_o, meta_data=None, data_dct=None):
    d_coef_matrix = get_d_coef_matrix(sigma_ns_h, sigma_ns_o, meta_data=meta_data, data_dct=data_dct)
    mat_rank = np.linalg.matrix_rank(d_coef_matrix)
    print(mat_rank, d_coef_matrix.shape)
    d_coef_matrix_shorter, d_idx_dropped = drop_dependent_columns(d_coef_matrix)
    return d_idx_dropped


###############################################################################
# Density calibration
###############################################################################
def optimize_density(d_len, d_to_f_mat, d_to_fh_mat, d_to_fh_total_mat, single_t_d_len, d_idx_dropped=None):
    """
    Calibrate the coarse preference density d.

    Parameters
    ----------
    d_len : int
        Total number of density variables across all hours.
    d_to_f_mat : np.ndarray, shape (2 * N_DATA * S, d_len)
        Linear map from density to equilibrium lane flows.
    d_to_fh_mat : np.ndarray, shape (C * N_DATA, d_len)
        Linear map from density to HOT-lane flow by occupancy.
    d_to_fh_total_mat : np.ndarray, shape (N_DATA, d_len)
        Linear map from density to total HOT-lane flow.
    single_t_d_len : int
        Number of coarse preference cells per hour.
    d_idx_dropped : list[int] or None
        Optional list of linearly dependent columns to force to zero.

    Returns
    -------
    obj_val : float
    density : np.ndarray, shape (d_len,)
    f_h_ret : np.ndarray, shape (C * N_DATA,)
    f_h_total_ret : np.ndarray, shape (N_DATA,)
    """
    global N_DATA, S, C, N_HOUR, TRAIN_IDX, FLOW_TARGET, PROFILE_DATE_MAP
    global RATIO_TARGET, N_DATES_TRAIN, RATIO_INDEX_TO_IGNORE
    
    HOURLY_DENSITY_ALLOWED_MASK = get_hourly_density_feasibility_mask(
        UNIQUE_HOUR_LST,
        beta_range_lst_full=BETA_RANGE_LST_FULL,
        gamma_range_dct_full=GAMMA_RANGE_DCT_FULL,
        beta_range_lst_am=BETA_RANGE_LST_AM,
        gamma_range_dct_am=GAMMA_RANGE_DCT_AM,
    )

    d_to_f_mat = np.asarray(d_to_f_mat, dtype=float)
    d_to_fh_mat = np.asarray(d_to_fh_mat, dtype=float)
    d_to_fh_total_mat = np.asarray(d_to_fh_total_mat, dtype=float)
    flow_target = np.asarray(FLOW_TARGET, dtype=float)

    model = gp.Model("density_calibration")
    model.Params.OutputFlag = 1
    model.Params.NumericFocus = 1

    d = model.addMVar(d_len, lb=0.0, vtype=GRB.CONTINUOUS, name="d")

    # Implied equilibrium flows
    f_equi = model.addMVar(2 * N_DATA * S, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    f_h_equi = model.addMVar(C * N_DATA, lb=0.0, vtype=GRB.CONTINUOUS, name="fh")
    f_h_total_equi = model.addMVar(N_DATA, lb=0.0, vtype=GRB.CONTINUOUS, name="fh_total")

    model.addConstr(d_to_f_mat @ d == f_equi)
    model.addConstr(d_to_fh_mat @ d == f_h_equi)
    model.addConstr(d_to_fh_total_mat @ d == f_h_total_equi)

    # Optional: fix dropped / dependent columns to zero
#    if d_idx_dropped is not None:
#        for d_idx in d_idx_dropped:
#            model.addConstr(d[d_idx] == 0.0)
    
    # Hour-specific feasibility mask
    for hour_idx in range(N_HOUR):
        for k in range(single_t_d_len):
            if not HOURLY_DENSITY_ALLOWED_MASK[hour_idx, k]:
                model.addConstr(d[hour_idx * single_t_d_len + k] == 0.0)

    # Density sums to 1 within each hour
    for hour_idx in range(N_HOUR):
        lo = hour_idx * single_t_d_len
        hi = (hour_idx + 1) * single_t_d_len
        model.addConstr(d[lo:hi].sum() == 1.0)

    # Flow-matching loss on training period
    ordinary_pred = f_equi[:(TRAIN_IDX * S)]
    ordinary_tgt = flow_target[:(TRAIN_IDX * S)]

    hot_pred = f_equi[(N_DATA * S):(N_DATA * S + TRAIN_IDX * S)]
    hot_tgt = flow_target[(N_DATA * S):(N_DATA * S + TRAIN_IDX * S)]

    obj_ordinary = ((ordinary_pred - ordinary_tgt) * (ordinary_pred - ordinary_tgt)).sum() / max(TRAIN_IDX, 1)
    obj_hot = ((hot_pred - hot_tgt) * (hot_pred - hot_tgt)).sum() / max(TRAIN_IDX, 1)

    objective = obj_ordinary + obj_hot * 49 #9.0
#    objective = objective * 12 #144

    # HOT-lane occupancy ratio fitting
    flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_equi

    for c in range(C):
        pred_c = PROFILE_DATE_MAP[:N_DATES_TRAIN, :TRAIN_IDX] @ f_h_equi[(c * N_DATA):(c * N_DATA + TRAIN_IDX)]
        tgt_c = RATIO_TARGET[:N_DATES_TRAIN, c] * flow_ratio_target_total[:N_DATES_TRAIN]
        ratio_loss = pred_c - tgt_c
        objective += (ratio_loss * ratio_loss).sum() / max(TRAIN_IDX, 1) * 1e1

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed to find an optimal solution. Status = {model.Status}")

    obj_val = float(model.ObjVal)
    density = d.X.copy()
    f_h_ret = f_h_equi.X.copy()
    f_h_total_ret = f_h_total_equi.X.copy()

    return obj_val, density, f_h_ret, f_h_total_ret
    
def calibrate_density():
    beta_lst, gamma_box_map, d_idx_start_lst = get_grid()
    segment_type_num = int(S * (S + 1) / 2)

    if N_CPU > 1:
        sigma_ns_h = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S), dtype=np.float32)
        sigma_ns_o = np.zeros((N_DATA, len(beta_lst), segment_type_num, C, S), dtype=np.float32)
        occ_frac_ns = np.zeros((N_DATA, len(beta_lst), segment_type_num, C), dtype=np.float32)
        batch_size = int(math.ceil(N_DATA / N_CPU))

        results = Parallel(n_jobs=N_CPU)(
            delayed(profile_given_data_single)(
                i * batch_size,
                min(N_DATA, (i + 1) * batch_size),
                beta_lst,
                gamma_box_map,
                segment_type_num
            ) for i in range(N_CPU)
        )

        for res in tqdm(results):
            sigma_ns_h += res[0]
            sigma_ns_o += res[1]
            occ_frac_ns += res[2]
    else:
        sigma_ns_h, sigma_ns_o, occ_frac_ns = profile_given_data_single(
            0, N_DATA, beta_lst, gamma_box_map, segment_type_num
        )

    d_idx_dropped = is_identifiable(sigma_ns_h, sigma_ns_o)

    single_t_d_len = len(d_idx_start_lst) - 1
    d_len = int(N_HOUR * single_t_d_len)

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
                            d_to_f_mat[relev_data_idx * S + s, hour_idx * single_t_d_len + d_idx] += (
                                1 / (c + 1)
                                * sigma_ns_o[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], segment_idx, c, s].sum(axis=1)
                                / elem_num
                                * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            )
                            d_to_f_mat[N_DATA * S + relev_data_idx * S + s, hour_idx * single_t_d_len + d_idx] += (
                                1 / (c + 1)
                                * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], segment_idx, c, s].sum(axis=1)
                                / elem_num
                                * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            )
                            d_to_fh_mat[c * N_DATA + relev_data_idx, hour_idx * single_t_d_len + d_idx] += (
                                1 / (c + 1)
                                * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], segment_idx, c, s].sum(axis=1)
                                / elem_num
                                * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            )
                            d_to_fh_total_mat[relev_data_idx, hour_idx * single_t_d_len + d_idx] += (
                                1 / (c + 1)
                                * sigma_ns_h[relev_data_idx, d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], segment_idx, c, s].sum(axis=1)
                                / elem_num
                                * HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]
                            )
                    segment_idx += 1

    if DENSITY_RETRAIN:
        obj_val, density, f_h_ret, f_h_total_ret = optimize_density(
            d_len, d_to_f_mat, d_to_fh_mat, d_to_fh_total_mat, single_t_d_len, d_idx_dropped
        )
    else:
        print("Loading density...")
        density = np.load("density/preference_density_general_updated.npy")
        f_h_ret = d_to_fh_mat @ density
        f_h_total_ret = d_to_fh_total_mat @ density

    f_equi_ret = d_to_f_mat @ density
    df_tmp = pd.DataFrame.from_dict({"Flow Equi": f_equi_ret, "Flow Target": FLOW_TARGET})
    df_tmp["Lane Type"] = LANE_TYPE_ALL
    df_tmp["Date"] = DATE_LST_ALL
    df_tmp["Hour"] = HOUR_LST_ALL
    df_tmp["Segment"] = SEGMENT_LST_ALL
    df_tmp.to_csv("tmp.csv", index=False)

    dct_ratio = {"Date": date_lst}
    flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_ret
    for c in range(C):
        dct_ratio[f"Equi {c}"] = PROFILE_DATE_MAP @ f_h_ret[(c * N_DATA):((c + 1) * N_DATA)]
        dct_ratio[f"Target {c}"] = RATIO_TARGET[:, c] * flow_ratio_target_total

    df_tmp_ratio = pd.DataFrame.from_dict(dct_ratio)
    df_tmp_ratio.to_csv("tmp_ratio.csv", index=False)
    return density


###############################################################################
# Diagnostics
###############################################################################
def describe_density(density, meta_data=None):
    global N_HOUR, S, C, BETA_RANGE_LST, GAMMA_RANGE_DCT, HOUR_OD_DEMAND, UNIQUE_HOUR_LST
    if meta_data is not None:
        N_HOUR = meta_data["N_HOUR"]
        S = meta_data["S"]
        C = meta_data["C"]
        BETA_RANGE_LST = meta_data["BETA_RANGE_LST"]
        GAMMA_RANGE_DCT = meta_data["GAMMA_RANGE_DCT"]
        HOUR_OD_DEMAND = meta_data["HOUR_OD_DEMAND"]
        UNIQUE_HOUR_LST = meta_data["UNIQUE_HOUR_LST"]

    beta_lst, gamma_box_map, d_idx_start_lst = get_grid(
        beta_range_lst=BETA_RANGE_LST,
        gamma_range_dct=GAMMA_RANGE_DCT
    )
    beta_gamma_range_lst = get_beta_gamma_range_lst(
        beta_range_lst=BETA_RANGE_LST,
        gamma_range_dct=GAMMA_RANGE_DCT
    )

    segment_type_num = int(S * (S + 1) / 2)
    segment_range_lst = []
    for s_o in range(S):
        for s_d in range(s_o, S):
            segment_range_lst.append(f"{segment_lst[s_o]} to {segment_lst[s_d]}")

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
    beta_lst, gamma_box_map, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)
    segment_pop = np.zeros(segment_type_num)

    for segment_type_idx in range(segment_type_num):
        density_idx_begin = hour_idx * single_t_d_len * segment_type_num + segment_type_idx
        density_idx_end = (hour_idx + 1) * single_t_d_len * segment_type_num
        pop = density[density_idx_begin:density_idx_end:segment_type_num].sum()
        segment_pop[segment_type_idx] = pop

    return segment_pop


def describe_segment_type_strategy(sigma, density, hour_idx, eps=1e-3):
    beta_lst, gamma_box_map, d_idx_start_lst = get_grid()
    single_t_d_len = len(d_idx_start_lst) - 1
    segment_type_num = int(S * (S + 1) / 2)

    segment_range_lst = []
    for s_o in range(S):
        for s_d in range(s_o, S):
            segment_range_lst.append(f"{segment_lst[s_o]} to {segment_lst[s_d]}")

    segment_pop = get_segment_pop(density, hour_idx)
    for segment_idx in range(segment_type_num):
        print(f"Segment {segment_range_lst[segment_idx]}:")
        for s in range(S):
            for c in range(C):
                sigma_o_idx = np.arange(
                    segment_idx * C * S * 2 + c * S * 2 + s * 2,
                    len(sigma),
                    segment_type_num * C * S * 2
                )
                sigma_o_total = sigma[sigma_o_idx].sum()
                sigma_h_total = sigma[sigma_o_idx + 1].sum()
                if sigma_o_total + sigma_h_total > eps:
                    print(f"\tS = {s}, C = {c + 1}: sigma_o = {sigma_o_total:.2f}, sigma_h = {sigma_h_total:.2f}")


###############################################################################
# Iterative equilibrium solver
###############################################################################
def get_flow_from_toll_iterative(
    density,
    tau_cs,
    meta_data=None,
    rho=0.25,
    hour_idx=12,
    num_itr=10,
    lam=1e-1,
    schedule_lst=None,
    eta=1
):
    global DISTANCE_ARR

    if schedule_lst is None:
        schedule_lst = []

    beta_lst_np, gamma_box_map_np, d_idx_start_lst = get_grid(
        beta_range_lst=BETA_RANGE_LST,
        gamma_range_dct=GAMMA_RANGE_DCT
    )
    single_t_d_len = len(d_idx_start_lst) - 1
    n_grids = len(beta_lst_np)
    segment_type_num = int(S * (S + 1) / 2)

    segment_type_strategy_len = segment_type_num * C * S * 2

    segment_type_strategy_to_flow_o_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_flow_h_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_flow_h2_map = np.zeros((S * C, segment_type_strategy_len))
    segment_type_strategy_to_agents_o_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_agents_h_map = np.zeros((S, segment_type_strategy_len))
    equi_profile_to_strategy_density_vec = np.zeros((len(beta_lst_np), segment_type_strategy_len))
    equi_profile_to_strategy_pop_vec = np.zeros((len(beta_lst_np), segment_type_strategy_len))
    segment_len_lst = np.zeros(segment_type_num)

    for c in range(C):
        segment_type_idx = 0
        for s_o in range(S):
            for s_d in range(s_o, S):
                demand = HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_type_idx]
                col_idx_o_begin = segment_type_idx * C * S * 2 + c * S * 2 + s_o * 2
                col_idx_o_end = segment_type_idx * C * S * 2 + c * S * 2 + (s_d + 1) * 2
                col_idx_h_begin = col_idx_o_begin + 1
                col_idx_h_end = col_idx_o_end + 1

                segment_type_strategy_to_flow_o_map[s_o:(s_d + 1), col_idx_o_begin:col_idx_o_end:2] += 1 / (c + 1) * demand
                segment_type_strategy_to_flow_h_map[s_o:(s_d + 1), col_idx_h_begin:col_idx_h_end:2] += 1 / (c + 1) * demand
                segment_type_strategy_to_flow_h2_map[(s_o * C + c):((s_d + 1) * C + c):C, col_idx_h_begin:col_idx_h_end:2] += 1 / (c + 1) * demand
                segment_type_strategy_to_agents_o_map[s_o:(s_d + 1), col_idx_o_begin:col_idx_o_end:2] += demand
                segment_type_strategy_to_agents_h_map[s_o:(s_d + 1), col_idx_h_begin:col_idx_h_end:2] += demand
                segment_len_lst[segment_type_idx] = s_d + 1 - s_o
                segment_type_idx += 1

    segment_density_lst = np.zeros(segment_type_num)
    loss_arr = []
    segment_type_strategy = np.zeros(segment_type_strategy_len)

    segment_type_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            demand = HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_type_idx]
            density_sum = density[(hour_idx * single_t_d_len):((hour_idx + 1) * single_t_d_len)].sum()
            segment_density_lst[segment_type_idx] = density_sum

            for s in range(s_o, s_d + 1):
                seg_start = segment_type_idx * C * S * 2
                begin = seg_start + s * 2
                seg_end = (segment_type_idx + 1) * C * S * 2
                o_idx_lst = np.arange(begin, seg_end, S * 2)
                h_idx_lst = o_idx_lst + 1

                if segment_density_lst[segment_type_idx] > 0:
                    segment_type_strategy[o_idx_lst] = 1 / (segment_len_lst[segment_type_idx] * C * 2)
                    segment_type_strategy[h_idx_lst] = 1 / (segment_len_lst[segment_type_idx] * C * 2)

                    if density_sum > 0:
                        for d_idx in range(single_t_d_len):
                            d_val = density[hour_idx * single_t_d_len + d_idx]
                            elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
                            equi_val = d_val / elem_num / density_sum / segment_len_lst[segment_type_idx]

                            equi_profile_to_strategy_density_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], o_idx_lst
                            ] = equi_val
                            equi_profile_to_strategy_density_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], h_idx_lst
                            ] = equi_val
                            equi_profile_to_strategy_pop_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], o_idx_lst
                            ] = equi_val * density_sum * demand
                            equi_profile_to_strategy_pop_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1], h_idx_lst
                            ] = equi_val * density_sum * demand
            segment_type_idx += 1

    o_lanes = int(NUM_LANES * (1 - rho))
    h_lanes = NUM_LANES - o_lanes
    utility_cost_arr = []

    tau_lst = np.zeros((1, segment_type_strategy_len))
    tau_lst[:, 1::2] = np.tile(tau_cs.reshape(C * S), segment_type_num)

    # bookkeeping only
    gamma_mid_map = gamma_midpoint_map_from_boxes(gamma_box_map_np)
    gamma_lst_c_long = np.tile(gamma_mid_map.repeat(S * 2, axis=1), reps=(1, segment_type_num))

    segment_type_strategy_to_flow_o_map_t = torch.from_numpy(segment_type_strategy_to_flow_o_map).float()
    segment_type_strategy_to_flow_h_map_t = torch.from_numpy(segment_type_strategy_to_flow_h_map).float()
    segment_type_strategy_t = torch.from_numpy(segment_type_strategy).float().requires_grad_()

    DISTANCE_ARR_t = torch.from_numpy(DISTANCE_ARR).float()
    beta_lst_t = torch.from_numpy(beta_lst_np).float()
    gamma_box_map_t = torch.from_numpy(gamma_box_map_np).float()
    tau_cs_t = torch.from_numpy(tau_cs).float()
    equi_profile_to_strategy_density_vec_t = torch.from_numpy(equi_profile_to_strategy_density_vec).float()
    
    for itr in tqdm(range(num_itr), leave=False):
        flow_o = segment_type_strategy_to_flow_o_map_t @ segment_type_strategy_t
        flow_h = segment_type_strategy_to_flow_h_map_t @ segment_type_strategy_t
        latency_o = get_cost(flow_o / o_lanes, DISTANCE_ARR_t)
        latency_h = get_cost(flow_h / h_lanes, DISTANCE_ARR_t)

        sigma_s_h, sigma_s_o, occ_frac = solve_sigma_given_parameters_vec_torch(
            beta_lst_t,
            gamma_box_map_t,
            latency_o,
            latency_h,
            tau_cs_t
        )

        sigma_s = torch.zeros((len(beta_lst_np), segment_type_strategy_len), dtype=beta_lst_t.dtype)
        sigma_s[:, ::2] += sigma_s_o.reshape((len(beta_lst_np), segment_type_strategy_len // 2))
        sigma_s[:, 1::2] += sigma_s_h.reshape((len(beta_lst_np), segment_type_strategy_len // 2))
        equi_profile = (equi_profile_to_strategy_density_vec_t * sigma_s).sum(dim=0)

        sq_loss = torch.sum((segment_type_strategy_t - equi_profile) ** 2)
        loss = sq_loss
        loss.backward()

        with torch.no_grad():
            segment_type_strategy_t -= lam * segment_type_strategy_t.grad

        loss_arr.append(float(loss.detach().cpu()))
        segment_type_strategy_t.grad.zero_()

        if itr in schedule_lst:
            lam *= eta

    segment_type_strategy = segment_type_strategy_t.detach().numpy()
    flow_o = segment_type_strategy_to_flow_o_map @ segment_type_strategy
    flow_h = segment_type_strategy_to_flow_h_map @ segment_type_strategy
    latency_o = get_cost(flow_o / o_lanes, DISTANCE_ARR)
    latency_h = get_cost(flow_h / h_lanes, DISTANCE_ARR)

    sigma_s_np = sigma_s.detach().numpy()
    equi_profile_pop = equi_profile_to_strategy_pop_vec * sigma_s_np
    agents_o = segment_type_strategy_to_agents_o_map @ segment_type_strategy
    agents_h = segment_type_strategy_to_agents_h_map @ segment_type_strategy
    total_travel_time = (agents_o * latency_o + agents_h * latency_h).sum()
    total_emission = (flow_o * latency_o + flow_h * latency_h).sum()
    total_revenue = (equi_profile_pop * tau_lst).sum()

    latency_tmp = np.zeros(S * 2)
    latency_tmp[::2] = latency_o
    latency_tmp[1::2] = latency_h
    latency_lst = np.tile(latency_tmp, segment_type_num * C).reshape((1, segment_type_strategy_len))
    total_utility_cost = (
        equi_profile_pop
        * (beta_lst_np.reshape((len(beta_lst_np), 1)) * latency_lst + tau_lst + gamma_lst_c_long)
    ).sum()

    flow_o_equi = flow_o
    flow_h_equi = segment_type_strategy_to_flow_h2_map @ segment_type_strategy

    return (
        segment_type_strategy,
        loss_arr,
        latency_o,
        latency_h,
        utility_cost_arr,
        total_travel_time,
        total_emission,
        total_revenue,
        total_utility_cost,
        flow_o_equi,
        flow_h_equi,
    )

def get_flow_from_toll_iterative_mann(
    density,
    tau_cs,
    meta_data=None,
    rho=0.25,
    hour_idx=12,
    num_itr=10,
    lam=0.5
):
    """
    Mann iteration adapted to the current pipeline.

    Uses:
      - coarse density on (beta, gamma)-cells
      - fine partition only in beta
      - occupancy fractions over each coarse gamma box
    """
    global DISTANCE_ARR, BETA_RANGE_LST, GAMMA_RANGE_DCT, N_HOUR, S, C, HOUR_OD_DEMAND

    if meta_data is not None:
        N_HOUR = meta_data["N_HOUR"]
        S = meta_data["S"]
        C = meta_data["C"]
        BETA_RANGE_LST = meta_data["BETA_RANGE_LST"]
        GAMMA_RANGE_DCT = meta_data["GAMMA_RANGE_DCT"]
        HOUR_OD_DEMAND = meta_data["HOUR_OD_DEMAND"]

    # ------------------------------------------------------------------
    # 1) Grid
    # ------------------------------------------------------------------
    beta_lst, gamma_box_map, d_idx_start_lst = get_grid(
        beta_range_lst=BETA_RANGE_LST,
        gamma_range_dct=GAMMA_RANGE_DCT
    )
    single_t_d_len = len(d_idx_start_lst) - 1
    n_grids = len(beta_lst)
    segment_type_num = int(S * (S + 1) / 2)

    # ------------------------------------------------------------------
    # 2) Auxiliary matrices
    # ------------------------------------------------------------------
    segment_type_strategy_len = segment_type_num * C * S * 2

    segment_type_strategy_to_flow_o_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_flow_h_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_flow_h2_map = np.zeros((S * C, segment_type_strategy_len))
    segment_type_strategy_to_agents_o_map = np.zeros((S, segment_type_strategy_len))
    segment_type_strategy_to_agents_h_map = np.zeros((S, segment_type_strategy_len))
    equi_profile_to_strategy_density_vec = np.zeros((len(beta_lst), segment_type_strategy_len))
    equi_profile_to_strategy_pop_vec = np.zeros((len(beta_lst), segment_type_strategy_len))
    segment_len_lst = np.zeros(segment_type_num)

    for c in range(C):
        segment_type_idx = 0
        for s_o in range(S):
            for s_d in range(s_o, S):
                demand = HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_type_idx]

                col_idx_o_begin = segment_type_idx * C * S * 2 + c * S * 2 + s_o * 2
                col_idx_o_end = segment_type_idx * C * S * 2 + c * S * 2 + (s_d + 1) * 2
                col_idx_h_begin = col_idx_o_begin + 1
                col_idx_h_end = col_idx_o_end + 1

                segment_type_strategy_to_flow_o_map[s_o:(s_d + 1), col_idx_o_begin:col_idx_o_end:2] = 1 / (c + 1) * demand
                segment_type_strategy_to_flow_h_map[s_o:(s_d + 1), col_idx_h_begin:col_idx_h_end:2] = 1 / (c + 1) * demand
                segment_type_strategy_to_flow_h2_map[(s_o * C + c):((s_d + 1) * C + c):C, col_idx_h_begin:col_idx_h_end:2] = 1 / (c + 1) * demand
                segment_type_strategy_to_agents_o_map[s_o:(s_d + 1), col_idx_o_begin:col_idx_o_end:2] = demand
                segment_type_strategy_to_agents_h_map[s_o:(s_d + 1), col_idx_h_begin:col_idx_h_end:2] = demand

                segment_len_lst[segment_type_idx] = s_d + 1 - s_o
                segment_type_idx += 1

    # ------------------------------------------------------------------
    # 3) Initial guess
    # ------------------------------------------------------------------
    segment_density_lst = np.zeros(segment_type_num)
    loss_arr = []
    utility_cost_arr = []

    segment_type_strategy = np.zeros(segment_type_strategy_len)

    segment_type_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            demand = HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_type_idx]
            density_sum = density[(hour_idx * single_t_d_len):((hour_idx + 1) * single_t_d_len)].sum()
            segment_density_lst[segment_type_idx] = density_sum

            for s in range(s_o, s_d + 1):
                seg_start = segment_type_idx * C * S * 2
                begin = seg_start + s * 2
                seg_end = (segment_type_idx + 1) * C * S * 2

                o_idx_lst = np.arange(begin, seg_end, S * 2)
                h_idx_lst = o_idx_lst + 1

                if segment_density_lst[segment_type_idx] > 0:
                    segment_type_strategy[o_idx_lst] = 1 / (segment_len_lst[segment_type_idx] * C * 2)
                    segment_type_strategy[h_idx_lst] = 1 / (segment_len_lst[segment_type_idx] * C * 2)

                    if density_sum > 0:
                        for d_idx in range(single_t_d_len):
                            d_val = density[hour_idx * single_t_d_len + d_idx]
                            elem_num = d_idx_start_lst[d_idx + 1] - d_idx_start_lst[d_idx]
                            equi_val = d_val / elem_num / density_sum / segment_len_lst[segment_type_idx]

                            equi_profile_to_strategy_density_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1],
                                o_idx_lst
                            ] = equi_val
                            equi_profile_to_strategy_density_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1],
                                h_idx_lst
                            ] = equi_val

                            equi_profile_to_strategy_pop_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1],
                                o_idx_lst
                            ] = equi_val * density_sum * demand
                            equi_profile_to_strategy_pop_vec[
                                d_idx_start_lst[d_idx]:d_idx_start_lst[d_idx + 1],
                                h_idx_lst
                            ] = equi_val * density_sum * demand

            segment_type_idx += 1

    # ------------------------------------------------------------------
    # 4) Iteration setup
    # ------------------------------------------------------------------
    o_lanes = int(NUM_LANES * (1 - rho))
    h_lanes = NUM_LANES - o_lanes

    tau_lst = np.zeros((1, segment_type_strategy_len))
    tau_lst[:, 1::2] = np.tile(tau_cs.reshape(C * S), segment_type_num)

    # For utility bookkeeping only
    gamma_mid_map = gamma_midpoint_map_from_boxes(gamma_box_map)   # (n_grids, C)
    gamma_lst_c_long = np.tile(
        gamma_mid_map.repeat(S * 2, axis=1),
        reps=(1, segment_type_num)
    )

    segment_type_strategy_prev = segment_type_strategy.copy()
    segment_type_strategy_best = segment_type_strategy.copy()
    loss_best = np.inf

    equi_profile_dens = None
    sigma_s = None

    # ------------------------------------------------------------------
    # 5) Mann iteration
    # ------------------------------------------------------------------
    for itr in tqdm(range(num_itr), leave=False):
        if itr < 50:
            lam = 1.0
        else:
            lam = 1e-2
        segment_type_strategy_v = (segment_type_strategy + itr * segment_type_strategy_prev) / (itr + 1)
        segment_type_strategy_prev = segment_type_strategy_v.copy()

        flow_o = segment_type_strategy_to_flow_o_map @ segment_type_strategy_v
        flow_h = segment_type_strategy_to_flow_h_map @ segment_type_strategy_v

        latency_o = get_cost(flow_o / o_lanes, DISTANCE_ARR)
        latency_h = get_cost(flow_h / h_lanes, DISTANCE_ARR)

        sigma_s_h, sigma_s_o, occ_frac = solve_sigma_given_parameters_vec(
            beta_lst,
            gamma_box_map,
            latency_o,
            latency_h,
            tau_cs
        )

        sigma_s = np.zeros((len(beta_lst), segment_type_strategy_len), dtype=np.float32)
        sigma_s[:, ::2] = sigma_s_o.reshape((len(beta_lst), segment_type_strategy_len // 2))
        sigma_s[:, 1::2] = sigma_s_h.reshape((len(beta_lst), segment_type_strategy_len // 2))

        equi_profile = (equi_profile_to_strategy_density_vec * sigma_s).sum(axis=0)

        loss = np.mean((segment_type_strategy - equi_profile) ** 2)
        segment_type_strategy = segment_type_strategy * (1 - lam) + equi_profile * lam
        loss_arr.append(loss)

        if loss < loss_best:
            loss_best = loss
            segment_type_strategy_best = segment_type_strategy.copy()

        latency_tmp = np.zeros(S * 2)
        latency_tmp[::2] = latency_o
        latency_tmp[1::2] = latency_h
        latency_lst = np.tile(latency_tmp, segment_type_num * C).reshape((1, segment_type_strategy_len))

        if itr > 0:
            total_utility_cost_prev = (
                equi_profile_dens
                * (beta_lst.reshape((len(beta_lst), 1)) * latency_lst + tau_lst + gamma_lst_c_long)
            ).sum()

            equi_profile_dens = equi_profile_to_strategy_density_vec * sigma_s

            total_utility_cost = (
                equi_profile_dens
                * (beta_lst.reshape((len(beta_lst), 1)) * latency_lst + tau_lst + gamma_lst_c_long)
            ).sum()

            utility_cost_arr.append(total_utility_cost_prev - total_utility_cost)
        else:
            equi_profile_dens = equi_profile_to_strategy_density_vec * sigma_s

        if loss < 1e-8:
            break

    # ------------------------------------------------------------------
    # 6) Final evaluation using best iterate
    # ------------------------------------------------------------------
    segment_type_strategy = segment_type_strategy_best

    flow_o = segment_type_strategy_to_flow_o_map @ segment_type_strategy
    flow_h = segment_type_strategy_to_flow_h_map @ segment_type_strategy
    latency_o = get_cost(flow_o / o_lanes, DISTANCE_ARR)
    latency_h = get_cost(flow_h / h_lanes, DISTANCE_ARR)

    # Recompute final best response at the selected iterate
    sigma_s_h, sigma_s_o, occ_frac = solve_sigma_given_parameters_vec(
        beta_lst,
        gamma_box_map,
        latency_o,
        latency_h,
        tau_cs
    )

    sigma_s = np.zeros((len(beta_lst), segment_type_strategy_len), dtype=np.float32)
    sigma_s[:, ::2] = sigma_s_o.reshape((len(beta_lst), segment_type_strategy_len // 2))
    sigma_s[:, 1::2] = sigma_s_h.reshape((len(beta_lst), segment_type_strategy_len // 2))

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

    total_utility_cost = (
        equi_profile_pop
        * (beta_lst.reshape((len(beta_lst), 1)) * latency_lst + tau_lst + gamma_lst_c_long)
    ).sum()

    flow_o_equi = flow_o
    flow_h_equi = segment_type_strategy_to_flow_h2_map @ segment_type_strategy

    return (
        segment_type_strategy,
        loss_arr,
        latency_o,
        latency_h,
        utility_cost_arr,
        total_travel_time,
        total_emission,
        total_revenue,
        total_utility_cost,
        flow_o_equi,
        flow_h_equi,
    )

###############################################################################
# Toll design
###############################################################################
def toll_design_grid_search_single(
    tau_tup_lst,
    density,
    hour_idx=12,
    rho_lst=[0.25, 0.5, 0.75],
    num_itr=100,
    lam=1e-2
):
    dct_results = {
        "Rho": [],
        "Loss": [],
        "Total Travel Time": [],
        "Total Emission": [],
        "Total Revenue": [],
        "Total Utility Cost": []
    }
    for s in range(S):
        dct_results[f"Toll {s}"] = []

    for tau_tup in tqdm(tau_tup_lst):
        tau_cs = np.zeros((C, S))
        tau_cs[0, :] = np.array(tau_tup)
        tau_cs[1, :] = tau_cs[0, :] / 4

        for rho in rho_lst:
            (
                segment_type_strategy,
                loss_arr,
                latency_o,
                latency_h,
                utility_cost_arr,
                total_travel_time,
                total_emission,
                total_revenue,
                total_utility_cost,
                _,
                _
            ) = get_flow_from_toll_iterative_mann(
                density,
                tau_cs=tau_cs,
                rho=rho,
                hour_idx=hour_idx,
                num_itr=num_itr,
                lam=lam
            )

            dct_results["Rho"].append(rho)
            dct_results["Loss"].append(np.min(loss_arr))
            dct_results["Total Travel Time"].append(total_travel_time)
            dct_results["Total Emission"].append(total_emission)
            dct_results["Total Revenue"].append(total_revenue)
            dct_results["Total Utility Cost"].append(total_utility_cost)

            for s in range(S):
                dct_results[f"Toll {s}"].append(tau_tup[s])

    return dct_results


def toll_design_grid_search(
    density,
    hour_idx=12,
    tau_max=5,
    d_tau=1,
    rho_lst=[0.25, 0.5, 0.75],
    num_itr=100,
    lam=1e-2
):
    dct_results = None
    tau_lst_single = np.linspace(0, tau_max, int(tau_max // d_tau) + 1)
    tau_tup_lst = list(itertools.product(*[tau_lst_single] * S))
    batch_size = int(math.ceil(len(tau_tup_lst) / N_CPU))

    results = Parallel(n_jobs=N_CPU)(
        delayed(toll_design_grid_search_single)(
            tau_tup_lst[(i * batch_size):min((i + 1) * batch_size, len(tau_tup_lst))],
            density,
            hour_idx,
            rho_lst,
            num_itr,
            lam
        )
        for i in range(N_CPU)
    )

    for res in results:
        if dct_results is None:
            dct_results = res
        else:
            for key in dct_results:
                dct_results[key] += res[key]

    return pd.DataFrame.from_dict(dct_results)

def toll_design_fine_tune_single(
    hour_idx_lst,
    tau_tup_lst,
    density,
    rho_lst=[0.25, 0.5, 0.75],
    num_itr=100,
    lam=1e-2
):
    assert len(hour_idx_lst) == len(tau_tup_lst)
    dct_results = {
        "Rho": [],
        "Loss": [],
        "Total Travel Time": [],
        "Total Emission": [],
        "Total Revenue": [],
        "Total Utility Cost": []
    }
    for s in range(S):
        dct_results[f"Toll {s}"] = []
    dct_results["Hour"] = []
    n_items = len(hour_idx_lst)

    for n in tqdm(range(n_items)):
        hour_idx, tau_tup = hour_idx_lst[n], tau_tup_lst[n]
        tau_cs = np.zeros((C, S))
        tau_cs[0, :] = np.array(tau_tup)
        tau_cs[1, :] = tau_cs[0, :] / 4

        for rho in rho_lst:
            (
                segment_type_strategy,
                loss_arr,
                latency_o,
                latency_h,
                utility_cost_arr,
                total_travel_time,
                total_emission,
                total_revenue,
                total_utility_cost,
                _,
                _
            ) = get_flow_from_toll_iterative_mann(
                density,
                tau_cs=tau_cs,
                rho=rho,
                hour_idx=hour_idx,
                num_itr=num_itr,
                lam=lam
            )

            dct_results["Rho"].append(rho)
            dct_results["Loss"].append(np.min(loss_arr))
            dct_results["Total Travel Time"].append(total_travel_time)
            dct_results["Total Emission"].append(total_emission)
            dct_results["Total Revenue"].append(total_revenue)
            dct_results["Total Utility Cost"].append(total_utility_cost)

            for s in range(S):
                dct_results[f"Toll {s}"].append(tau_tup[s])
            dct_results["Hour"].append(hour_idx + 7)

    return dct_results

###############################################################################
# Main
###############################################################################
if DENSITY_RECALIBRATE:
    density = calibrate_density()
    if DENSITY_RETRAIN:
        np.save("density/preference_density_general_updated.npy", density)
else:
    density = np.load("density/preference_density_general_updated.npy")

#describe_density(density)
#assert False

"""
hour_idx = 7
segment_type_strategy, loss_arr, latency_o, latency_h, utility_cost_arr, total_travel_time, total_emission, total_revenue, total_utility_cost, flow_o_equi, flow_h_equi  = get_flow_from_toll_iterative(density, tau_cs = np.array([[1, 0.25, 0], [2, 0.5, 0], [2.5, 0.625, 0], [4.0, 1, 0], [5, 1.25, 0]]).T, rho = 0.25, hour_idx = hour_idx, num_itr = 5000, lam = 1e-3)
#segment_type_strategy, loss_arr, latency_o, latency_h, utility_cost_arr, total_travel_time, total_emission, total_revenue, total_utility_cost, flow_o_equi, flow_h_equi  = get_flow_from_toll_iterative_mann(density, tau_cs = np.array([[0.5, 0.125, 0], [1.5, 0.375, 0], [1, 0.25, 0], [1.5, 0.375, 0], [0.5, 0.125, 0]]).T, rho = 0.25, hour_idx = hour_idx, num_itr = 500, lam = 1e-2)
print(segment_type_strategy.round(3))
print(segment_type_strategy.sum())
print(total_travel_time, total_emission, total_revenue, total_utility_cost)
print("Final Loss:", loss_arr[-1])
print(flow_o_equi)
print(flow_h_equi)
describe_segment_type_strategy(segment_type_strategy, density, hour_idx = hour_idx, eps = 1e-2)

plt.plot(loss_arr)
#plt.yscale("log")
plt.title(f"loss = {np.min(loss_arr):.2e}")
plt.savefig("loss.png")
plt.clf()
plt.close()

assert False
"""

fname = "toll_design_multiseg_hour=16-17.csv" #"toll_design_multiseg.csv"
rho_lst = [0.25, 0.50] #[0.25, 0.50, 0.75]

if not FINE_TUNE:
    df_all = None
    for hour_idx in [9, 10]: #tqdm(range(12)):
        df_res = toll_design_grid_search(
            density,
            hour_idx=hour_idx,
            tau_max=5,
            d_tau=0.5,
            rho_lst=rho_lst,#[0.25],
            num_itr=250,
            lam=1
        )
        df_res["Hour"] = hour_idx + 7
        if df_all is None:
            df_all = df_res
        else:
            df_all = pd.concat([df_all, df_res], ignore_index=True)
else:
    df_all = pd.read_csv(fname)
    df_sub = df_all[df_all["Loss"] > 1e-6]
    hour_idx_lst = (df_sub["Hour"] - 7).values.tolist()
    tau_tup_lst = df_sub[[x for x in df_sub.columns if x.startswith("Toll")]].values.tolist()
    tau_tup_lst = [tuple(x) for x in tau_tup_lst]
    batch_size = int(math.ceil(len(tau_tup_lst) / N_CPU))

    results = Parallel(n_jobs=N_CPU)(
        delayed(toll_design_fine_tune_single)(
            hour_idx_lst[(i * batch_size):min((i + 1) * batch_size, len(hour_idx_lst))],
            tau_tup_lst[(i * batch_size):min((i + 1) * batch_size, len(tau_tup_lst))],
            density,
            rho_lst,
            200,
            1e-2
        )
        for i in range(N_CPU)
    )
    dct_results = None
    for res in results:
        if dct_results is None:
            dct_results = res
        else:
            for key in dct_results:
                dct_results[key] += res[key]

    df_sub = pd.DataFrame.from_dict(dct_results)
    df_all = pd.concat([df_all, df_sub]).drop_duplicates(subset = ["Rho", "Hour"] + [f"Toll {s}" for s in range(S)], keep="last")
df_all.to_csv(fname, index=False)
