import json
import math
import itertools
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import scipy
from scipy import optimize
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, csr_array, dia_matrix, vstack
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
from hdbscan import HDBSCAN
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from tqdm import tqdm

## Script Options
N_CPU = 2
DENSITY_RECALIBRATE = True
DENSITY_RETRAIN = True
TRAIN_FRAC = 0.8#0.8

## Hyperparameters
NUM_LANES = 4
BPR_POWER = 4
BPR_A = 7e-4 #2.4115e-13
BPR_B = 0.7906
DISTANCE = 7.16 # miles
WINDOW_SIZE = 5 #15

C = 3
S = 5

## Load Data
### Date, Hour, Segment, HOV Flow, Ordinary Flow, HOV Travel Time, Ordinary Travel Time, Avg_total_toll
df = pd.read_csv("data/df_meta.csv") #pd.read_csv("hourly_demand_20210401.csv")
# df = df[df["Segment"] == "3460 - Hesperian/238 NB"]
df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df = df.dropna()
df = df[(df["Date"] >= "2021-02-01") & (df["Date"] <= "2021-05-31")]
df = df[(df["Hour"] >= 14) & (df["Hour"] <= 18)]


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
            HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx] = df_od_demand[(df_od_demand["Hour"] == hour) & (df_od_demand["Origin"] == origin_seg) & (df_od_demand["Destination"] == dest_seg)].iloc[0]["Demand"]
            segment_idx += 1
###
#N_DATES = len(df["Date"].unique())
## N_DATES, N_DATA, S
## Days to ignore: 3/31, 4/23, 4/26, 6/30
RATIO_INDEX_TO_IGNORE = [22, 39, 40, 86]
DATES_TO_IGNORE = ["2021-02-15", "2021-03-31", "2021-04-23", "2021-04-26", "2021-06-30"]
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
TRAIN_TEST = np.zeros(N_DATA)
TRAIN_TEST[:TRAIN_IDX] = 1

# ===========================================================
# USER SETTINGS
# ===========================================================

# beta and gamma bounds
BETA_MIN, BETA_MAX = 0.0, 5.0
GAMMA2_MIN, GAMMA2_MAX = 0.0, 4.0
GAMMA3_MIN, GAMMA3_MAX = 0.0, 2.0

BETA_RANGE_LST = [(0, 0.1), (0.1, 2), (4, 5)]
GAMMA_RANGE_DCT = {
    1: [(0, 0)],
    2: [(0, 0.1), (2, 4)],
    3: [(0, 0.1), (1, 2)]
}

DELTA = 0.25   # cube side length

# model dimensions
C = 3
T = LATENCY_O_LST.shape[0]
S = LATENCY_O_LST.shape[1]


# ===========================================================
# STEP 1 — compute best response at cube midpoints
# ===========================================================

def compute_best_response(beta, gamma_vec):
    """
    Computes best_c and lane choices for EVERY OD pair (s_o,s_d)
    using CUMULATIVE gamma (consistent with original grid code).

    Output signature structure:
        best_c_od  : (T, segment_type_num)
        lane_od    : (T, C, S, segment_type_num)
    """

    T = TRAIN_IDX
    C, S = TAU_CS_LST.shape[1], TAU_CS_LST.shape[2]
    segment_type_num = int(S * (S + 1) / 2)

    # -------------------------------------------------------
    # make gamma cumulative across classes (same as old code)
    # -------------------------------------------------------
    gamma_cum = np.cumsum(gamma_vec)
    gamma_cum = gamma_cum.reshape(1, C, 1)   # (1,C,1)

    # ---- lane costs per record, class, segment ----
    cost_o = beta * LATENCY_O_LST[:T, None, :]       # (T,C,S) via broadcast

    cost_h = (
        beta * LATENCY_HOV_LST[:T, None, :]
        + TAU_CS_LST[:T, :, :]
        + gamma_cum                                 # <-- cumulative gamma
    )

    # lane decision per segment
    lane_choice = (cost_h < cost_o).astype(np.int8)  # (T,C,S)

    # total cost on chosen lane per segment
    total_cost_seg = lane_choice * cost_h + (1 - lane_choice) * cost_o

    # -------------------------------------------------------
    # compute OD-pair costs and best-c for each (s_o,s_d)
    # -------------------------------------------------------
    best_c_od = np.zeros((T, segment_type_num), dtype=np.int8)
    lane_od   = np.zeros((T, C, S, segment_type_num), dtype=np.int8)

    seg_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):

            # path cost = sum segments along [s_o, s_d]
            path_cost = total_cost_seg[:, :, s_o:(s_d+1)].sum(axis=2)   # (T,C)

            # best occupancy class for this OD pair
            best_c = np.argmin(path_cost, axis=1).astype(np.int8)      # (T,)

            best_c_od[:, seg_idx] = best_c

            # store associated lane choices for the chosen class
            for t in range(T):
                c_star = best_c[t]
                lane_od[t, c_star, s_o:(s_d+1), seg_idx] = \
                    lane_choice[t, c_star, s_o:(s_d+1)]

            seg_idx += 1

    # -------------------------------------------------------
    # HASHABLE SIGNATURE (for clustering)
    # -------------------------------------------------------
    sig = (
        best_c_od.tobytes(),   # OD-pair best-c
        lane_od.tobytes()      # associated lane choices
    )

    return sig



# ===========================================================
# STEP 2 — cluster cubes with identical behavior
# ===========================================================

def neighbor_graph(cube_idxs):
    coords = np.array([cubes[idx][:3] for idx in cube_idxs], int)
    mapping = {tuple(c):m for m,c in enumerate(coords)}

    rows, cols, data = [], [], []
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for m,c in enumerate(coords):
        for dx,dy,dz in neigh:
            nb = (c[0]+dx, c[1]+dy, c[2]+dz)
            if nb in mapping:
                rows.append(m); cols.append(mapping[nb]); data.append(1)

    A = csr_matrix((data,(rows,cols)),shape=(len(coords),len(coords)))
    return A

def behavior_distance(sig1, sig2):
    b1 = np.frombuffer(sig1[0], np.int8)
    l1 = np.frombuffer(sig1[1], np.int8)
    b2 = np.frombuffer(sig2[0], np.int8)
    l2 = np.frombuffer(sig2[1], np.int8)
    return (
        np.mean(b1 != b2) +
        np.mean(l1 != l2)
    )

def flatten_signature(sig):
    """
    Convert signature from compute_best_response_od
    into a flat binary vector suitable for clustering.
    """
    best_c = np.frombuffer(sig[0], np.int8)
    lanes  = np.frombuffer(sig[1], np.int8)
    return np.concatenate([best_c, lanes])


def cluster_to_K(cubes, signatures, K):
    """
    Behavior-driven clustering into exactly K clusters
    using agglomerative clustering with Hamming distance.

    Returns list of clusters compatible with the rest of your pipeline.
    """

    # ---- flatten behavior signatures ----
    X = np.vstack([flatten_signature(sig) for sig in signatures])

    # ---- Agglomerative clustering (Hamming metric) ----
    model = AgglomerativeClustering(
        n_clusters=K,
        metric="manhattan",   # == Hamming for 0/1 or int8
        linkage="average"
    )

    labels = model.fit_predict(X)

    # ---- collect cubes per cluster ----
    groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[lab].append(idx)

    clusters = []

    for lab, idx_list in groups.items():

        pts = [cubes[i] for i in idx_list]

        betas  = np.array([p[3] for p in pts])
        gammas = np.stack([p[4] for p in pts])

        cluster = {
            "points": pts,
            "count": len(pts),
            "midpoint": (
                betas.mean(),
                gammas.mean(axis=0)
            ),
            "signature": signatures[idx_list[0]]  # representative
        }

        clusters.append(cluster)

    return clusters

def cluster_by_lsh_signature(cubes, signatures, sample_frac=0.02):
    """
    Lightweight LSH-style near-duplicate detector.
    Groups cubes whose signatures match on a small random subset of bits.
    """

    buckets = defaultdict(list)

    for i, sig in enumerate(signatures):
        best_c = np.frombuffer(sig[0], np.int8)
        lanes  = np.frombuffer(sig[1], np.int8)

        vec = np.concatenate([best_c, lanes])

        m = max(1, int(len(vec) * sample_frac))
        idx = np.random.choice(len(vec), m, replace=False)

        # bucket key = sampled bit pattern
        key = tuple(vec[idx])
        buckets[key].append(i)

    # Now treat each bucket as a cluster (or refine inside if needed)
    clusters = []
    for idx_list in buckets.values():

        pts = [cubes[i] for i in idx_list]
        betas  = np.array([p[3] for p in pts])
        gammas = np.stack([p[4] for p in pts])

        clusters.append({
            "points": pts,
            "count": len(pts),
            "midpoint": (betas.mean(), gammas.mean(axis=0)),
            "signature": signatures[idx_list[0]]
        })

    return clusters

def cubes_to_debug_clusters(
    cubes,
    signatures,
    beta_ranges=BETA_RANGE_LST,
    gamma_ranges=GAMMA_RANGE_DCT,
):
    from collections import defaultdict

    idx_per_bucket = defaultdict(list)

    for idx, cube in enumerate(cubes):
        i, j, k, beta_mid, gamma_vec = cube

        gamma_raw = gamma_vec.copy()

        # ---- beta bucket ----
        beta_bucket = None
        for b_lo, b_hi in beta_ranges:
            if b_lo <= beta_mid <= b_hi:
                beta_bucket = (b_lo, b_hi)
                break
        if beta_bucket is None:
            continue

        # ---- gamma buckets ----
        bucket_key = [beta_bucket]

        ok_all = True
        for c in range(2, len(gamma_raw)+1):
            g = gamma_raw[c-1]

            matching = [r for r in gamma_ranges[c]
                        if r[0] <= g <= r[1]]

            if not matching:
                ok_all = False
                break

            bucket_key.append(matching[0])

        if not ok_all:
            continue

        idx_per_bucket[tuple(bucket_key)].append(idx)

    # ---- build clusters ----
    clusters = []
    for _, idx_list in idx_per_bucket.items():
        pts = [cubes[i] for i in idx_list]

        betas  = np.array([p[3] for p in pts])
        gammas = np.stack([p[4] for p in pts])

        clusters.append({
            "points": pts,
            "count": len(pts),
            "midpoint": (betas.mean(), gammas.mean(axis=0)),
            "signature": signatures[idx_list[0]]
        })

    return clusters



# ===========================================================
# STEP 3 — visualization
# ===========================================================

def visualize_clusters_3d_colored(clusters):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    for c,cl in enumerate(clusters):
        pts = cl["points"]
        betas = np.array([p[3] for p in pts])
        g2 = np.array([p[4][1] for p in pts])
        g3 = np.array([p[4][2] for p in pts])
        ax.scatter(betas, g2, g3, s=6, label=f"C{c}")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma_2$")
    ax.set_zlabel(r"$\gamma_3$")
#    ax.legend(loc="upper left", fontsize=8)
    plt.title("Behavioral Clusters (Tolerance Grouped)")
    plt.savefig("behavior_clusters.png")
    plt.clf()
    plt.close()

def find_cluster_boundaries(clusters):
    """
    For each cluster, identify boundary points:
    points that have at least one 6-connected neighbor
    not belonging to the same cluster.
    """

    # convenience: map every cube index -> cluster id
    cube_to_cluster = {}

    for cid, cl in enumerate(clusters):
        for p in cl["points"]:
            i,j,k,_,_ = p
            cube_to_cluster[(i,j,k)] = cid

    boundaries = defaultdict(list)

    # 6-connected neighborhood in the cube grid
    neighbors = [(1,0,0),(-1,0,0),
                 (0,1,0),(0,-1,0),
                 (0,0,1),(0,0,-1)]

    for cid, cl in enumerate(clusters):
        for p in cl["points"]:
            i,j,k,beta,gamma = p
            is_boundary = False

            for dx,dy,dz in neighbors:
                nb = (i+dx, j+dy, k+dz)

                # if neighbor missing or belongs to another cluster → boundary
                if nb not in cube_to_cluster or cube_to_cluster[nb] != cid:
                    is_boundary = True
                    break

            if is_boundary:
                boundaries[cid].append((beta, gamma[1], gamma[2]))

    return boundaries

def find_cluster_vertices(clusters, tol=1e-9):
    """
    Compute geometric vertices for each cluster, automatically
    handling degenerate cases (planar / linear / singleton clusters).

    Returns
    -------
    dict : cluster_id -> list[(beta, gamma2, gamma3)]
    """

    vertices = defaultdict(list)

    for cid, cl in enumerate(clusters):
        pts = []
        for p in cl["points"]:
            beta = p[3]
            g2   = p[4][1]
            g3   = p[4][2]
            pts.append((beta, g2, g3))

        pts = np.array(pts)
        if pts.shape[0] == 0:
            continue

        # ---- detect intrinsic dimension ----
        span = pts.max(axis=0) - pts.min(axis=0)
        varying = span > tol
        dim = varying.sum()

        # ===============================
        # 0-D: single unique point
        # ===============================
        if dim == 0:
            vertices[cid] = [tuple(pts[0])]
            continue

        # ===============================
        # 1-D: points lie on a line
        # ===============================
        if dim == 1:
            # get the coordinate that varies
            k = np.where(varying)[0][0]
            order = np.argsort(pts[:,k])
            endpoints = [tuple(pts[order[0]]),
                         tuple(pts[order[-1]])]
            vertices[cid] = endpoints
            continue

        # ===============================
        # 2-D: points lie in a plane
        # ===============================
        if dim == 2:
            sub = pts[:, varying]          # project to 2D plane
            if len(np.unique(sub, axis=0)) < 3:
                # degenerate polygon → endpoints
                order = np.lexsort(sub.T)
                endpoints = [tuple(pts[order[0]]),
                             tuple(pts[order[-1]])]
                vertices[cid] = endpoints
            else:
                hull2 = ConvexHull(sub)
                verts2 = sub[hull2.vertices]
                # lift back to 3D by combining with fixed coord(s)
                lifted = []
                for v in verts2:
                    full = np.zeros(3)
                    full[varying] = v
                    full[~varying] = pts[0, ~varying]
                    lifted.append(tuple(full))
                vertices[cid] = lifted
            continue

        # ===============================
        # 3-D: full convex hull
        # ===============================
        hull3 = ConvexHull(pts)
        verts3 = pts[hull3.vertices]
        vertices[cid] = [tuple(v) for v in verts3]

    return vertices

def cluster_param_ranges(clusters):
    """
    For each cluster, compute min/max for:
        beta, gamma2, gamma3

    Returns
    -------
    ranges : dict
        cid -> {
            "beta_min": float, "beta_max": float,
            "gamma2_min": float, "gamma2_max": float,
            "gamma3_min": float, "gamma3_max": float,
            "count": int
        }
    """

    ranges = defaultdict(dict)

    for cid, cl in enumerate(clusters):
        pts = cl["points"]
        if len(pts) == 0:
            continue

        betas  = np.array([p[3]     for p in pts])
        g2s    = np.array([p[4][1]  for p in pts])
        g3s    = np.array([p[4][2]  for p in pts])

        ranges[cid] = {
            "beta_min":  betas.min(),
            "beta_max":  betas.max(),
            "gamma2_min": g2s.min(),
            "gamma2_max": g2s.max(),
            "gamma3_min": g3s.min(),
            "gamma3_max": g3s.max(),
            "count": len(pts)
        }

    return ranges

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
    for data_idx in range(n_data):
        for grid_idx in range(n_grids):
            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    best_c = best_c_lst[segment_idx][data_idx, grid_idx]
                    lane_cs_h[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    lane_cs_o[data_idx,grid_idx, segment_idx, best_c,s_o:(s_d+1)] = 1 - lane_cs[data_idx,grid_idx,best_c,s_o:(s_d+1)]
                    segment_idx += 1
    return lane_cs_h, lane_cs_o #lane_cs[:,best_c,:]

def profile_given_data_clusters(
    beta_vec,          # shape (K,)
    gamma_vec_c,       # shape (K, C)
    segment_type_num,
    latency_o_lst,     # (N_DATA, S)
    latency_hov_lst,   # (N_DATA, S)
    tau_cs_lst,        # (N_DATA, C, S)
    n_jobs=1
):
    """
    Compute sigma profiles for a batch of (beta, gamma) parameter sets.

    Parameters
    ----------
    beta_vec : array (K,)
    gamma_vec_c : array (K, C)
    segment_type_num : int
    latency_o_lst : (N_DATA, S)
    latency_hov_lst : (N_DATA, S)
    tau_cs_lst : (N_DATA, C, S)
    n_jobs : parallel workers over data dimension

    Returns
    -------
    sigma_ns_h : (N_DATA, K, segment_type_num, C, S)
    sigma_ns_o : (N_DATA, K, segment_type_num, C, S)
    """

    N_DATA, S = latency_o_lst.shape
    C = gamma_vec_c.shape[1]
    K = beta_vec.shape[0]

    sigma_ns_h = np.zeros((N_DATA, K, segment_type_num, C, S))
    sigma_ns_o = np.zeros((N_DATA, K, segment_type_num, C, S))

    # ------------------------------------------------------
    # worker for a block of data rows
    # ------------------------------------------------------
    def _worker(lo, hi):
        block_h = np.zeros((hi-lo, K, segment_type_num, C, S))
        block_o = np.zeros((hi-lo, K, segment_type_num, C, S))

        for local_i, data_i in enumerate(range(lo, hi)):
            sh, so = solve_sigma_given_parameters_vec(
                beta_vec,
                gamma_vec_c,
                latency_o_lst[data_i, :],
                latency_hov_lst[data_i, :],
                tau_cs_lst[data_i, :, :]
            )
            # returned shapes: (1, K, seg_type, C, S)
            block_h[local_i] = sh[0]
            block_o[local_i] = so[0]

        return block_h, block_o

    # ------------------------------------------------------
    # parallel over data index
    # ------------------------------------------------------
    batch = int(math.ceil(N_DATA / n_jobs))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(
            i*batch,
            min(N_DATA, (i+1)*batch)
        )
        for i in range(n_jobs)
    )

    # scatter back into global buffers
    offset = 0
    for bh, bo in results:
        n = bh.shape[0]
        sigma_ns_h[offset:offset+n] = bh
        sigma_ns_o[offset:offset+n] = bo
        offset += n

    return sigma_ns_h, sigma_ns_o

def compute_cluster_sigma_profiles(clusters,
                                   segment_type_num,
                                   latency_o_lst,
                                   latency_hov_lst,
                                   tau_cs_lst,
                                   n_jobs=1):

    K = len(clusters)
    N_DATA = latency_o_lst.shape[0]

    sigma_ns_h = np.zeros((N_DATA, K, segment_type_num, C, S))
    sigma_ns_o = np.zeros((N_DATA, K, segment_type_num, C, S))

    for k in tqdm(range(len(clusters))):
        cl = clusters[k]

        # all cube midpoints in this cluster
        betas  = np.array([p[3] for p in cl["points"]])
        gammas = np.stack([p[4] for p in cl["points"]])   # (M, C)

        # compute sigma for each cube separately
        sigma_h_k, sigma_o_k = profile_given_data_clusters(
            betas,
            gammas,
            segment_type_num,
            latency_o_lst,
            latency_hov_lst,
            tau_cs_lst,
            n_jobs=n_jobs
        )

        # aggregate across cubes in the cluster (mean profile)
        sigma_ns_h[:, k, :, :, :] = sigma_h_k.mean(axis=1)
        sigma_ns_o[:, k, :, :, :] = sigma_o_k.mean(axis=1)

    return sigma_ns_h, sigma_ns_o

def calibrate_density_clusters(clusters, n_jobs=1):
    """
    Given behavioral clusters in (beta, gamma) space,
    assign one demand density parameter per cluster (per hour),
    and solve for densities to match observed flows.

    clusters: list of dict, each with key "midpoint" = (beta, gamma_vec)
              where gamma_vec has length C (e.g., [0, gamma2, gamma3])
    """
    global FLOW_TARGET, RATIO_TARGET, FLOW_COEF

    segment_type_num = int(S * (S + 1) / 2)
    K = len(clusters)

    # ---------------------------------------------
    # build parameter arrays from cluster midpoints
    # ---------------------------------------------
    beta_clusters = np.zeros(K)
    gamma_clusters_c = np.zeros((K, C))
    for k, cl in enumerate(clusters):
        beta_k, gamma_vec_k = cl["midpoint"]
        beta_clusters[k] = beta_k
        # ensure length C
        if len(gamma_vec_k) != C:
            raise ValueError("gamma_vec length does not match C")
        gamma_clusters_c[k, :] = gamma_vec_k

    # ---------------------------------------------
    # compute sigma profile for each cluster
    # ---------------------------------------------
    sigma_ns_h, sigma_ns_o = compute_cluster_sigma_profiles(
        clusters,
        segment_type_num,
        LATENCY_O_LST,
        LATENCY_HOV_LST,
        TAU_CS_LST,
        n_jobs=n_jobs
    )

    # Optional: identifiability check can be adapted if desired
    # is_identifiable(sigma_ns_h, sigma_ns_o, meta_data=meta_data, data_dct=data_dct)

    # ---------------------------------------------
    # build linear map from densities to flows
    # ---------------------------------------------
    # d indices: (hour, cluster)
    single_t_d_len = K
    d_len = int(N_HOUR * single_t_d_len)

    # equilibrium flows
    # ordinary lanes: (N_DATA, S) -> flattened to length N_DATA * S
    # hov: (N_DATA, S, C) -> flattened to length N_DATA * S * C
    d_to_fo_mat = np.zeros((N_DATA * S, d_len))
    d_to_fh_mat = np.zeros((N_DATA * S * C, d_len))

    for hour_idx in tqdm(range(N_HOUR), desc="Building d_to_f matrices"):
        t = UNIQUE_HOUR_LST[hour_idx]
        relev_data_idx = np.where(HOUR_LST == t)[0]

        for k in range(K):
            d_col = hour_idx * single_t_d_len + k

            segment_idx = 0
            for s_o in range(S):
                for s_d in range(s_o, S):
                    # OD demand for this hour and OD pair
                    od_dem = HOUR_OD_DEMAND[hour_idx * segment_type_num + segment_idx]

                    for s in range(s_o, s_d + 1):
                        for c in range(C):
                            # sigma_ns_o / sigma_ns_h shape:
                            # (N_DATA, K, segment_type_num, C, S)
                            sigma_o_slice = sigma_ns_o[relev_data_idx, k, segment_idx, c, s]
                            sigma_h_slice = sigma_ns_h[relev_data_idx, k, segment_idx, c, s]

                            # contribution to ordinary flow (aggregation over occupancy)
                            d_to_fo_mat[relev_data_idx * S + s, d_col] += \
                                1.0 / (c + 1) * sigma_o_slice * od_dem

                            # contribution to hov flow
                            d_to_fh_mat[relev_data_idx * S * C + s * C + c, d_col] += \
                                1.0 / (c + 1) * sigma_h_slice * od_dem

                    segment_idx += 1

    # ---------------------------------------------
    # build and solve Gurobi model
    # ---------------------------------------------
    model = gp.Model("density_clusters")

    # ==================================================
    # Decision variables
    # ==================================================
    d = model.addMVar(d_len, lb=0.0, vtype=GRB.CONTINUOUS, name="d")
    f_o_equi = model.addMVar(N_DATA * S,     lb=0.0, vtype=GRB.CONTINUOUS, name="f_o")
    f_h_equi = model.addMVar(N_DATA * S * C, lb=0.0, vtype=GRB.CONTINUOUS, name="f_h")

    # linear mapping from densities to flows
    model.addConstr(d_to_fo_mat @ d == f_o_equi, name="fo_link")
    model.addConstr(d_to_fh_mat @ d == f_h_equi, name="fh_link")
    
    for hour_idx in range(N_HOUR):
        density_expr = gp.LinExpr(0.0)
        for k in range(single_t_d_len):
            d_col = hour_idx * single_t_d_len + k
            density_expr += d[d_col]
        model.addConstr(density_expr == 1)

    # ------------------------------------------------------------------
    # FLOW-MATCHING LOSS
    # Mirrors:
    # objective = ((f_equi[:(2*TRAIN_IDX*S)] - FLOW_TARGET[:...])
    #              * FLOW_COEF[:...] * ... ).sum() / TRAIN_IDX
    # but without concatenating MVars.
    # ------------------------------------------------------------------
    flow_obj = gp.QuadExpr()

    # Ordinary part: indices 0 .. TRAIN_IDX*S-1
    for s in range(S):
        for n in range(TRAIN_IDX):
            idx_o = s * N_DATA + n                  # matches FLOW_O_LST construction
            diff_o = f_o_equi[idx_o] - float(FLOW_TARGET[idx_o])
            w_o = float(FLOW_COEF[idx_o])
            flow_obj += (w_o * diff_o) * (w_o * diff_o)

    # HOV part: next TRAIN_IDX*S entries in FLOW_TARGET
    # HOV flow here is total across classes for each (n,s)
    for s in range(S):
        for n in range(TRAIN_IDX):
            # total HOV flow for (n,s) across classes
            hov_expr = gp.LinExpr(0.0)
            for c in range(C):
                idx_hc = n * (S * C) + s * C + c    # (data, segment, class)
                hov_expr += f_h_equi[idx_hc]

            idx_h = N_DATA * S + s * N_DATA + n     # HOV portion in FLOW_TARGET
            diff_h = hov_expr - float(FLOW_TARGET[idx_h])
            w_h = float(FLOW_COEF[idx_h])

            flow_obj += (w_h * diff_h) * (w_h * diff_h)

    flow_obj /= TRAIN_IDX


    # ------------------------------------------------------------------
    # RATIO LOSS
    # Mirrors the logic:
    #
    # flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_equi
    # ratio_loss = PROFILE_DATE_MAP[:N_DATES_TRAIN,:TRAIN_IDX] @ f_h_equi[...]
    #              - RATIO_TARGET[:N_DATES_TRAIN,c] * flow_ratio_target_total[:N_DATES_TRAIN]
    # ------------------------------------------------------------------
    ratio_obj = gp.QuadExpr()

    # 1) total HOV flow per data index (sum over segments & classes)
    f_h_total_equi = [gp.LinExpr(0.0) for _ in range(N_DATA)]
    for n in range(N_DATA):
        expr = gp.LinExpr(0.0)
        for s in range(S):
            for c in range(C):
                idx = n * (S * C) + s * C + c
                expr += f_h_equi[idx]
        f_h_total_equi[n] = expr

    # 2) flow_ratio_target_total[date] = sum_{n} PROFILE_DATE_MAP[date,n] * f_h_total_equi[n]
    flow_ratio_target_total = [gp.LinExpr(0.0) for _ in range(N_DATES)]
    for d_idx in range(N_DATES):
        expr = gp.LinExpr(0.0)
        for n in range(N_DATA):
            coef = PROFILE_DATE_MAP[d_idx, n]
            if coef != 0:
                expr += float(coef) * f_h_total_equi[n]
        flow_ratio_target_total[d_idx] = expr

    # Optional helper (kept for parity; not used as constraint)
    all_seg_flow = 0.0
    for s in range(S):
        all_seg_flow += FLOW_TARGET[(N_DATA * S + s)::S]
    daily_flow_lb = PROFILE_DATE_MAP @ all_seg_flow  # purely numeric

    # 3) class-specific HOV flow per data index (sum over segments only)
    f_h_class_data = [[gp.LinExpr(0.0) for _ in range(N_DATA)] for _ in range(C)]
    for c in range(C):
        for n in range(N_DATA):
            expr = gp.LinExpr(0.0)
            for s in range(S):
                idx = n * (S * C) + s * C + c
                expr += f_h_equi[idx]
            f_h_class_data[c][n] = expr

    # 4) ratio penalty per class and date (training dates only)
    ratio_idx = [i for i in range(len(date_lst)) if i not in RATIO_INDEX_TO_IGNORE]

    for c in range(C):
        for d_local, d_idx in enumerate(ratio_idx):
            if d_local >= N_DATES_TRAIN:
                break

            # aggregate equilibrium class-c flow for this date
            agg_flow_expr = gp.LinExpr(0.0)
            for n in range(TRAIN_IDX):
                coef = PROFILE_DATE_MAP[d_idx, n]
                if coef != 0:
                    agg_flow_expr += float(coef) * f_h_class_data[c][n]

            # target flow: RATIO_TARGET[d_idx,c] * flow_ratio_target_total[d_idx]
            target_expr = float(RATIO_TARGET[d_idx, c]) * flow_ratio_target_total[d_idx]

            diff = agg_flow_expr - target_expr
            ratio_obj += diff * diff

    ratio_obj /= TRAIN_IDX
    ratio_obj *= 10.0

    # ==================================================
    # FINAL OBJECTIVE
    # ==================================================
    obj = flow_obj + ratio_obj
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # ---------------------------------------------
    # extract densities and fitted flows
    # ---------------------------------------------
    density = np.array([d[i].X for i in range(d_len)])
    f_o_equi_ret = d_to_fo_mat @ density

    # hov flows per (data, seg, class)
    f_h_equi_ret = d_to_fh_mat @ density
    f_h_ret = f_h_equi_ret.reshape(N_DATA, S, C)

    # collapse classes to match original FLOW_TARGET layout
    f_h_total_ret_seg = f_h_ret.sum(axis=2).ravel()

    # final equilibrium vector (same length as FLOW_TARGET)
    f_equi_ret = np.concatenate([f_o_equi_ret, f_h_total_ret_seg])

    LANE_TYPE_ALL = ["Ordinary Lane"] * (N_DATA * S) + ["HOT Lane"] * (N_DATA * S)

    df_tmp = pd.DataFrame({
        "Flow Equi": f_equi_ret,
        "Flow Target": FLOW_TARGET,
        "Lane Type": LANE_TYPE_ALL,
        "Hour": HOUR_LST_ALL,
        "Segment": SEGMENT_LST_ALL,
    })
    df_tmp.to_csv("tmp.csv", index=False)


    # ---------------------------------------------
    # tmp_ratio.csv
    # ---------------------------------------------
    # reshape using known layout: (data, segment, class)
    f_h_ret = f_h_equi_ret.reshape(N_DATA, S, C)

    # total hov flow per record (sum over classes & segments)
    f_h_total_ret = f_h_ret.sum(axis=(1, 2))

    # total hov target flow per date
    flow_ratio_target_total = PROFILE_DATE_MAP @ f_h_total_ret

    dct_ratio = {"Date": date_lst}

    for c in range(C):

        # class-c hov flow per record (sum over segments)
        f_h_class_c = f_h_ret[:, :, c].sum(axis=1)

        # aggregate to date level
        dct_ratio[f"Equi {c}"] = PROFILE_DATE_MAP @ f_h_class_c

        # target flow for class-c
        dct_ratio[f"Target {c}"] = (
            RATIO_TARGET[:, c] * flow_ratio_target_total
        )

    df_tmp_ratio = pd.DataFrame(dct_ratio)
    df_tmp_ratio.to_csv("tmp_ratio.csv", index=False)

    return density


# build cubic grid
print("Building cubic grid...")
beta_vals  = np.arange(BETA_MIN,  BETA_MAX,  DELTA)
gamma2_vals = np.arange(GAMMA2_MIN, GAMMA2_MAX, DELTA)
gamma3_vals = np.arange(GAMMA3_MIN, GAMMA3_MAX, DELTA)

cubes = []
signatures = []

for i,b in enumerate(beta_vals):
    for j,g2 in enumerate(gamma2_vals):
        for k,g3 in enumerate(gamma3_vals):

            beta_mid  = b + DELTA/2
            gamma_vec = np.array([0.0, g2 + DELTA/2, g3 + DELTA/2])

            sig = compute_best_response(beta_mid, gamma_vec)

            cubes.append((i,j,k,beta_mid,gamma_vec))
            signatures.append(sig)

print("Clustering...")
#clusters = cluster_to_K(cubes, signatures, K=300)
clusters = cubes_to_debug_clusters(cubes, signatures)

print("Number of clusters:", len(clusters))
assert False

# call one of these:
#visualize_clusters_3d_colored(clusters)

#ranges = cluster_param_ranges(clusters)

#for cid, r in ranges.items():
#    print(f"\n=== Cluster {cid} (count={r['count']}) ===")
#    print(f" beta  ∈ [{r['beta_min']:.3f},  {r['beta_max']:.3f}]")
#    print(f" gamma2∈ [{r['gamma2_min']:.3f}, {r['gamma2_max']:.3f}]")
#    print(f" gamma3∈ [{r['gamma3_min']:.3f}, {r['gamma3_max']:.3f}]")

density = calibrate_density_clusters(
    clusters,
    n_jobs=N_CPU   # or 1
)
