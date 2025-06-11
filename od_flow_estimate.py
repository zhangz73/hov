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
        for i in tqdm(range(1, 13), leave = False):
            data_filename = "data/raw/d04_text_station_hour_2021_" + str(i).zfill(2) + ".txt"
            df_flow = pd.read_csv(data_filename, header = None, names = names)

#            df_flow = df_flow.dropna(axis="columns", how = 'all')
            df_flow = df_flow.loc[df_flow["Freeway"] == 880]
            df_flow = df_flow.loc[df_flow["Station"].isin(segment_dct[segment])]
            df_flow = df_flow.loc[df_flow["LaneType"].isin(["ML", "OR", "FR"])]
            # df_flow = df_flow.loc[df_flow["% Observed"] >= 75]
            df_flow = df_flow.loc[df_flow["Direction"] == 'N']
            df_flow = df_flow.fillna(0)


            df_flow["Time"] = pd.to_datetime(df_flow["Timestamp"])
            df_flow["Date"] = df_flow["Time"].dt.date
            df_flow["Hour"] = df_flow["Time"].dt.hour

            df_flow = df_flow[["Date", "Hour", "LaneType", "Total Flow"]]

            
            df_flow = df_flow.groupby(["Date", "Hour", "LaneType"]).mean().reset_index()
            df_flow["Segment"] = segment
            
            with open('data/df_PeMs_laneTypes.csv','a') as output_file:
                df_flow.to_csv(output_file, header=should_header, index=False)
                should_header = False
    #            print("wrote lines to output file:", len(df_flow), "from file", data_filename)
            
df_pems = pd.read_csv("data/df_PeMs_laneTypes.csv")
print(df_pems)

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

for hour_idx in range(N_HOURS):
    hour = HOUR_LST[hour_idx]
    for s in range(S):
        target_vec[hour_idx * S * 3 + s] = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["In Flow"]
        target_vec[hour_idx * S * 3 + S + s] = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["Main Flow"]
        target_vec[hour_idx * S * 3 + S * 2 + s] = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s])].iloc[0]["Out Flow"]
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

def max_entropy_gurobi():
    model = gp.Model()
    #model.Params.NonConvex = 2  # Allow log(T_ij)
    demand_len = N_HOURS * segment_type_num
    total_demand = model.addVars(demand_len, lb = 0, vtype = GRB.CONTINUOUS, name = "d")
    demand_log = model.addVars(demand_len, name = "logd")
    for i in range(demand_len):
        model.addGenConstrLog(total_demand[i], demand_log[i])
    #for row in range(constraint_mat.shape[0]):
    #    model.addConstr(gp.quicksum(constraint_mat[row, i] * total_demand[i] for i in range(demand_len)) == target_vec[row])
    #model.addConstr(constraint_mat @ total_demand == target_vec)
    objective = -gp.quicksum(total_demand[i] * demand_log[i] for i in range(demand_len))
    for row in range(constraint_mat.shape[0]):
        objective += -(gp.quicksum(constraint_mat[row, i] * total_demand[i] for i in range(demand_len)) - target_vec[row]) * (gp.quicksum(constraint_mat[row, i] * total_demand[i] for i in range(demand_len)) - target_vec[row]) * 10
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    obj_val = model.ObjVal
    demand_ret = np.zeros(demand_len)
    for i in range(demand_len):
        demand_ret[i] = total_demand[i].x
    return demand_ret

def max_entropy_analytical():
    demand_len = N_HOURS * segment_type_num
    total_demand = np.zeros(demand_len)
    for hour_idx in range(N_HOURS):
        hour = HOUR_LST[hour_idx]
        segment_idx = 0
        for s_o in range(S):
            origin_inflow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_o])].iloc[0]["In Flow"]
            total_outflow = 0
            for s_d in range(s_o, S):
                total_outflow += df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_d])].iloc[0]["Out Flow"]
            for s_d in range(s_o, S):
                dest_outflow = df_pems[(df_pems["Hour"] == hour) & (df_pems["Segment"] == SEGMENT_LST[s_d])].iloc[0]["Out Flow"]
                demand_idx = hour_idx * segment_type_num + segment_idx
                total_demand[demand_idx] = origin_inflow * (dest_outflow / total_outflow)
                segment_idx += 1
    return total_demand

def bertsimas_n_yan():
    pass

total_demand = max_entropy_analytical()
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
