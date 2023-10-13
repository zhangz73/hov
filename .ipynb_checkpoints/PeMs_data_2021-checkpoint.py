import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

RELEVANT_STATIONS = [400488, 401561, 400611, 400928, 400284, 400041, 408133, 408135, 417665, 412637, 417666, 408134, 400685, 401003, 400898, 400275, 400939, 400180, 400529, 400990, 400515, 400252]
# RELEVANT_STATIONS = [400488, 400611, 400284, 400041, 412637, 417666, 400275, 400990, 400515, 400252]

N = 8
# lane_names = list(itertools.chain(*[[f"Lane {i} Samples", f"Lane {i} Flow", f"Lane {i} Avg Occ", f"Lane {i} Avg Speed", f"Lane {i} Observed"] for i in range(N)]))
lane_names = list(itertools.chain(*[[f"Lane {i} Flow", f"Lane {i} Avg Occ", f"Lane {i} Avg Speed"] for i in range(N)]))
names = ["Timestamp", "Station", "District", "Freeway", "Direction", "LaneType", "StationLength", "Samples", "% Observed", "Total Flow", "Avg Occupancy", "Avg Speed", "Delay (V_t=35)", "Delay (V_t=40)", "Delay (V_t=45)", "Delay (V_t=50)", "Delay (V_t=55)", "Delay (V_t=60)"] + lane_names
# df_flow = pd.read_csv("data/d04_text_station_hour_2021_01.txt", header = None, names = names)


os.remove('data/df_PeMs.csv')
for i in range(1, 13):
    data_filename = "data/d04_text_station_hour_2021_" + str(i).zfill(2) + ".txt"
    df_flow = pd.read_csv(data_filename, header = None, names = names)

    df_flow = df_flow.dropna(axis="columns", how = 'all') 
    df_flow = df_flow.loc[df_flow["Freeway"] == 880]
    df_flow = df_flow.loc[df_flow["Station"].isin(RELEVANT_STATIONS)]
    df_flow = df_flow.loc[df_flow["LaneType"] == 'ML']
    # df_flow = df_flow.loc[df_flow["% Observed"] >= 75]
    df_flow = df_flow.loc[df_flow["Direction"] == 'N']
    df_flow = df_flow.fillna(0)

    df_flow["HOV Flow"] = df_flow["Lane 0 Flow"]
    
    df_flow["HOV Travel Time"] = df_flow["StationLength"]/df_flow["Lane 0 Avg Speed"]*60

    df_flow["Ordinary Cum Speed"] = 0

    # df_flow["Ordinary Flow"] = df_flow["Total Flow"]- df_flow["Lane 0 Flow"]
    # for j in range(1,7):
    #      df_flow["Ordinary Cum Speed"] +=  df_flow["Lane " + str(j) + " Avg Speed"]* df_flow["Lane " + str(j) + " Flow"]
    # df_flow["Ordinary Avg Speed"] = df_flow["Ordinary Cum Speed"]/df_flow["Ordinary Flow"]

    df_flow["Ordinary Flow"] = 0
    for j in range(1,4):
         df_flow["Ordinary Cum Speed"] +=  df_flow["Lane " + str(j) + " Avg Speed"]
         df_flow["Ordinary Flow"] += df_flow["Lane " + str(j) + " Flow"]
    df_flow["Ordinary Avg Speed"] = df_flow["Ordinary Cum Speed"]/3

    df_flow["Ordinary Travel Time"] = df_flow["StationLength"]/df_flow["Ordinary Avg Speed"]*60


    df_flow["Time"] = pd.to_datetime(df_flow["Timestamp"])
    df_flow["Date"] = df_flow["Time"].dt.date
    df_flow["Hour"] = df_flow["Time"].dt.hour

    df_flow = df_flow[["Date", "Hour", "HOV Flow", "Ordinary Flow","HOV Travel Time","Ordinary Travel Time"]]

    
    df_flow = df_flow.groupby(["Date","Hour"]).agg({"HOV Flow": "mean", "Ordinary Flow": "mean", "HOV Travel Time": "sum", "Ordinary Travel Time": "sum"}).reset_index()
    
    with open('data/df_PeMs.csv','a') as output_file:
        should_header = i == 1
        df_flow.to_csv(output_file, header=should_header, index=False)
        print("wrote lines to output file:", len(df_flow), "from file", data_filename)
        
