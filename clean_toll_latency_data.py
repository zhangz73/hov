import json
import math
import itertools
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

TRAIN_FRAC = 0.8#0.8

## Hyperparameters
NUM_LANES = 4
BPR_POWER = 4
BPR_A = 7e-4 #2.4115e-13
BPR_B = 0.7906
WINDOW_SIZE = 5 #15

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
TRAIN_TEST = np.zeros(N_DATA)
TRAIN_TEST[:TRAIN_IDX] = 1
