import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm

RELOAD_DATA = False

#RELEVANT_STATIONS = [400488, 401561, 400611, 400928, 400284, 400041, 408133, 408135, 417665, 412637, 417666, 408134, 400685, 401003, 400898, 400275, 400939, 400180, 400529, 400990, 400515, 400252]
# RELEVANT_STATIONS = [400488, 400611, 400284, 400041, 412637, 417666, 400275, 400990, 400515, 400252]
df_station = pd.read_csv("data/station_meta.csv")
df_station = df_station.dropna(subset = ["Segment"])
segment_dct = {}
segment_lst = list(df_station["Segment"].unique())
for segment in segment_lst:
    relv_stations = list(df_station[df_station["Segment"] == segment]["ID"])
    segment_dct[segment] = [int(x) for x in relv_stations]

if RELOAD_DATA:
    N = 8
    # lane_names = list(itertools.chain(*[[f"Lane {i} Samples", f"Lane {i} Flow", f"Lane {i} Avg Occ", f"Lane {i} Avg Speed", f"Lane {i} Observed"] for i in range(N)]))
    lane_names = list(itertools.chain(*[[f"Lane {i} Flow", f"Lane {i} Avg Occ", f"Lane {i} Avg Speed"] for i in range(N)]))
    names = ["Timestamp", "Station", "District", "Freeway", "Direction", "LaneType", "StationLength", "Samples", "% Observed", "Total Flow", "Avg Occupancy", "Avg Speed", "Delay (V_t=35)", "Delay (V_t=40)", "Delay (V_t=45)", "Delay (V_t=50)", "Delay (V_t=55)", "Delay (V_t=60)"] + lane_names
    # df_flow = pd.read_csv("data/d04_text_station_hour_2021_01.txt", header = None, names = names)

    #os.remove('data/df_PeMs.csv')
    should_header = True
    for segment in tqdm(segment_dct):
        for i in tqdm(range(2, 5), leave = False):
            data_filename = "data/raw/d04_text_station_hour_2021_" + str(i).zfill(2) + ".txt"
            df_flow = pd.read_csv(data_filename, header = None, names = names)

            df_flow = df_flow.dropna(axis="columns", how = 'all')
            df_flow = df_flow.loc[df_flow["Freeway"] == 880]
            df_flow = df_flow.loc[df_flow["Station"].isin(segment_dct[segment])]
            df_flow = df_flow.loc[df_flow["LaneType"].isin(["ML", "OR", "FR"])]
            # df_flow = df_flow.loc[df_flow["% Observed"] >= 75]
            df_flow = df_flow.loc[df_flow["Direction"] == 'N']
            df_flow = df_flow.fillna(0)


            df_flow["Time"] = pd.to_datetime(df_flow["Timestamp"])
            df_flow["Date"] = df_flow["Time"].dt.date
            df_flow["Hour"] = df_flow["Time"].dt.hour
            df_flow["HOT Flow"] = df_flow["Lane 0 Flow"]
            df_flow["Ordinary Flow"] = 0
            for j in range(1,4):
                df_flow["Ordinary Flow"] += df_flow["Lane " + str(j) + " Flow"]

            df_flow = df_flow[["Date", "Hour", "LaneType", "Ordinary Flow", "HOT Flow"]]

            
            df_flow = df_flow.groupby(["Date", "Hour", "LaneType"]).mean().reset_index()
            df_flow["Segment"] = segment
            
            with open('data/df_PeMs_laneTypes.csv','a') as output_file:
                df_flow.to_csv(output_file, header=should_header, index=False)
                should_header = False
    #            print("wrote lines to output file:", len(df_flow), "from file", data_filename)
            
df_pems = pd.read_csv("data/df_PeMs_laneTypes.csv")
print(df_pems)

df_meta = pd.read_csv("data/df_meta.csv")
df_pems = df_pems.merge(df_meta[["Date", "Hour", "Avg_total_toll"]], on = ["Date", "Hour"])
df_pems = df_pems.dropna()

df_pop = pd.read_csv("pop_fraction.csv", thousands = ",")
df_pop["Date"] = pd.to_datetime(df_pop["Date"]).dt.strftime("%Y-%m-%d")
df_pems = df_pems.merge(df_pop, on = "Date")
df_pems["Total"] = df_pems["Single"] + df_pems["TwoPeople"] + df_pems["ThreePlus"]
df_pems["Total Flow"] = (df_pems["Ordinary Flow"] + df_pems["HOT Flow"]) * (1 * df_pems["Single"] / df_pems["Total"] + 2 * df_pems["TwoPeople"] / df_pems["Total"] + 3 * df_pems["ThreePlus"] / df_pems["Total"])

### Compute in-flow and out-flow
df_pems = df_pems[["Hour", "Segment", "LaneType", "Total Flow"]].groupby(["Hour", "Segment", "LaneType"]).mean().reset_index()
df_pems = df_pems.pivot(index = ["Hour", "Segment"], columns = "LaneType", values = "Total Flow")
df_pems = df_pems.reset_index()
df_pems = df_pems.rename_axis(None, axis=1)
print(df_pems)
df_pems["In Flow"] = df_pems["ML"] * (df_pems["Segment"] == "3420 - Auto Mall N") + df_pems["OR"]
df_pems["Main Flow"] = df_pems["ML"] + df_pems["OR"] * (df_pems["Segment"] == "3420 - Auto Mall N") + df_pems["FR"] * (df_pems["Segment"] == "3460 - Hesperian/238 NB")
df_pems["Out Flow"] = df_pems["ML"] * (df_pems["Segment"] == "3460 - Hesperian/238 NB") + df_pems["FR"]
df_pems.to_csv("data/df_PeMs_FullLanes.csv", index = False)

### Calibrate total demand
N_HOURS = len(df_pems["Hour"].unique())
HOUR_LST = sorted(list(df_pems["Hour"].unique()))
SEGMENT_LST = ['3420 - Auto Mall N', '3430 - Mowry NB', '3440 - Decoto/84 NB', '3450 - Whipple NB', '3460 - Hesperian/238 NB']
S = len(SEGMENT_LST)
segment_type_num = int(S * (S + 1) / 2)
demand_len = N_HOURS * segment_type_num
constraint_mat = np.zeros((N_HOURS * S * 3, demand_len))
target_vec = np.zeros(N_HOURS * S * 3)
demand_vec = np.zeros(N_HOURS)
demand_scale_vec = np.zeros(N_HOURS)

## Adjust the imbalance in flow
for hour_idx in range(N_HOURS):
    hour = HOUR_LST[hour_idx]
    total_demand_in = 0
    total_demand_out = 0
    for s in range(S):
        in_flow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["In Flow"]
        out_flow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["Out Flow"]
        total_demand_in += in_flow
        total_demand_out += out_flow
    scale_factor = total_demand_out / total_demand_in
#    df_pems.loc[df_pems["Hour"] == hour, "In Flow"] *= scale_factor
#    df_pems.loc[(df_pems["Hour"] == hour) & (df_pems["Segment"] == "3420 - Auto Mall N"), "Main Flow"] *= scale_factor
    demand_scale_vec[hour_idx] = scale_factor

for hour_idx in range(N_HOURS):
    hour = HOUR_LST[hour_idx]
    for s in range(S):
        in_flow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["In Flow"]
        out_flow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["Out Flow"]
        target_vec[hour_idx * S * 3 + s] = in_flow
        target_vec[hour_idx * S * 3 + S + s] = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["Main Flow"]
        target_vec[hour_idx * S * 3 + S * 2 + s] = out_flow
    segment_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            demand_idx = hour_idx * segment_type_num + segment_idx
            ## In-flow constraints
            constraint_mat[hour_idx * S * 3 + s_o, demand_idx] = 1
            ## Main-flow constraints
            constraint_mat[hour_idx * S * 3 + S + s_o:(s_d+1), demand_idx] = 1
            ## Out-flow constraints
            constraint_mat[hour_idx * S * 3 + S * 2 + s_d, demand_idx] = 1
            segment_idx += 1

#def max_entropy_gurobi():
#    model = gp.Model()
#    #model.Params.NonConvex = 2  # Allow log(T_ij)
#    demand_len = N_HOURS * segment_type_num
#    total_demand = model.addVars(demand_len, lb = 0, vtype = GRB.CONTINUOUS, name = "d")
#    demand_log = model.addVars(demand_len, name = "logd")
#    for i in range(demand_len):
#        model.addGenConstrLog(total_demand[i], demand_log[i])
#    #for row in range(constraint_mat.shape[0]):
#    #    model.addConstr(gp.quicksum(constraint_mat[row, i] * total_demand[i] for i in range(demand_len)) == target_vec[row])
#    #model.addConstr(constraint_mat @ total_demand == target_vec)
#    objective = -gp.quicksum(total_demand[i] * demand_log[i] for i in range(demand_len))
#    for row in range(constraint_mat.shape[0]):
#        objective += -(gp.quicksum(constraint_mat[row, i] * total_demand[i] for i in range(demand_len)) - target_vec[row]) * (gp.quicksum(constraint_mat[row, i] * total_demand[i] for i in range(demand_len)) - target_vec[row]) * 10
#    model.setObjective(objective, GRB.MAXIMIZE)
#    model.optimize()
#    obj_val = model.ObjVal
#    demand_ret = np.zeros(demand_len)
#    for i in range(demand_len):
#        demand_ret[i] = total_demand[i].x
#    return demand_ret

def max_entropy_gurobi(penalty_weight=10.0, min_flow=1e-6):
    """
    Max-entropy OD estimation with flow constraints enforced via L1 slacks.
    Reference: "The most likely trip matrix estimated from traffic counts", Henk J. Van Zuylen, Luis G. Willumsen

    Problem:
        max  sum_i (-d_i * log d_i - d_i) - penalty_weight * sum_r (s_r^+ + s_r^-)
        s.t. A_r * d + s_r^+ - s_r^- = target_r   for all rows r
             d_i >= min_flow
             s_r^+, s_r^- >= 0

    Returns:
        demand_ret: np.array of shape (demand_len,)
    """
    demand_len = N_HOURS * segment_type_num
    num_rows   = constraint_mat.shape[0]

    model = gp.Model("max_entropy_od_l1")
    model.Params.NonConvex = 2  # needed for log general constraints

    # ------------------------------
    # Variables
    # ------------------------------
    # OD flows
    dvars = model.addMVar(demand_len, lb=min_flow, vtype=GRB.CONTINUOUS, name="d")

    # log(d) variables for entropy term
    logd = model.addMVar(demand_len, vtype=GRB.CONTINUOUS, name="logd")

    # Slack variables per constraint row (positive and negative)
    s_pos = model.addMVar(num_rows, lb=0.0, vtype=GRB.CONTINUOUS, name="s_pos")
    s_neg = model.addMVar(num_rows, lb=0.0, vtype=GRB.CONTINUOUS, name="s_neg")

    # ------------------------------
    # logd[i] = log(dvars[i])   (scalar form)
    # ------------------------------
    for i in range(demand_len):
        model.addGenConstrLog(dvars[i], logd[i], name=f"log_link_{i}")

    # ------------------------------
    # A_r * d + s_r^+ - s_r^- = target_r
    # ------------------------------
    row_exprs = []
    for r in range(num_rows):
        expr = gp.LinExpr()
        row = constraint_mat[r, :]
        for j, coef in enumerate(row):
            if coef != 0.0:
                expr += coef * dvars[j]
        # store expr if you want to inspect later
        row_exprs.append(expr)

        model.addConstr(
            expr + s_pos[r] - s_neg[r] == float(target_vec[r]),
            name=f"flow_balance_{r}"
        )

    # ------------------------------
    # Objective: entropy - L1 penalty on slacks
    # ------------------------------
    entropy_expr = gp.LinExpr()
    for i in range(demand_len):
        entropy_expr += - dvars[i] * logd[i] - dvars[i]

    slack_penalty = penalty_weight * (s_pos.sum() + s_neg.sum())

    obj = entropy_expr - slack_penalty
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Gurobi terminated with status {model.Status}")

    demand_ret = np.zeros(demand_len)
    for i in range(demand_len):
        demand_ret[i] = dvars[i].X

    # Optional diagnostics:
    max_resid = 0.0
    for r in range(num_rows):
        resid = s_pos[r].X - s_neg[r].X
        max_resid = max(max_resid, abs(resid))
    print("Max signed residual in constraints (via slacks):", max_resid)

    return demand_ret


def max_entropy_analytical():
    demand_len = N_HOURS * segment_type_num
    total_demand = np.zeros(demand_len)
    for hour_idx in range(N_HOURS):
        hour = HOUR_LST[hour_idx]
        segment_idx = 0
        hour_demand_scale = demand_scale_vec[hour_idx]
        for s_o in range(S):
            origin_inflow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_o])].iloc[0]["In Flow"]
            total_outflow = 0
            for s_d in range(s_o, S):
                total_outflow += df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_d])].iloc[0]["Out Flow"]
            for s_d in range(s_o, S):
                dest_outflow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_d])].iloc[0]["Out Flow"]
                total_inflow = 0
                for s_o in range(s_d + 1):
                    total_inflow += df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_o])].iloc[0]["In Flow"]
                demand_idx = hour_idx * segment_type_num + segment_idx
                total_demand[demand_idx] = origin_inflow * (dest_outflow / total_outflow) * hour_demand_scale #(origin_inflow * (dest_outflow / total_outflow) + (origin_inflow / total_inflow) * dest_outflow) / 2 #
                segment_idx += 1
    return total_demand

def bertsimas_n_yan():
    pass

total_demand = max_entropy_gurobi(penalty_weight=10.0, min_flow=1e-6) #
hour_lst_ret = []
origin_lst_ret = []
dest_lst_ret = []
demand_lst_ret = []
for hour_idx in range(N_HOURS):
    hour = HOUR_LST[hour_idx]
    segment_idx = 0
    for s_o in range(S):
        for s_d in range(s_o, S):
            demand_idx = hour_idx * segment_type_num + segment_idx
            demand_ret = total_demand[demand_idx]
            hour_lst_ret.append(hour)
            origin_lst_ret.append(SEGMENT_LST[s_o])
            dest_lst_ret.append(SEGMENT_LST[s_d])
            demand_lst_ret.append(demand_ret)
            segment_idx += 1
origin_lst_ret = ["3420 - Auto Mall NB" if x == "3420 - Auto Mall N" else x for x in origin_lst_ret]
dest_lst_ret = ["3420 - Auto Mall NB" if x == "3420 - Auto Mall N" else x for x in dest_lst_ret]
df_demand = pd.DataFrame.from_dict({"Hour": hour_lst_ret, "Origin": origin_lst_ret, "Destination": dest_lst_ret, "Demand": demand_lst_ret})
df_demand.to_csv("data/od_demand.csv", index = False)
