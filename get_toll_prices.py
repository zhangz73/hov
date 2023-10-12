import numpy as np
import pandas as pd

RELEVANT_ZONE = "3460 - Hesperian/238 NB"
RELEVANT_STATIONS = [400488, 401561, 400611, 400928, 400284, 400041, 408133, 408135, 417665, 412637, 417666, 408134, 400685, 401003, 400898, 400275, 400939, 400180, 400529, 400990, 400515, 400252]

## Load toll price data
df_all = None
for fname in ["NB_012021-032021.csv", "NB_042021-062021.csv", "NB_072021-092021.csv", "NB_102021-122021.csv"]:
    df = pd.read_csv(fname)
    #df = df[pd.to_numeric(df["Segment_Toll"], errors='coerce').notnull()]
    df = df[pd.to_numeric(df["Zone_Toll"], errors='coerce').notnull()]
    df = df.dropna(subset = ["Zone_Toll"])[["dtMsgStartTime2", "siZoneID", "iPlazaID", "Zone_Toll", "Segment_Toll"]]
    df = df.fillna(0)
    df = df[df["siZoneID"] == RELEVANT_ZONE]
    df["Date"] = pd.to_datetime(df["dtMsgStartTime2"]).dt.strftime("%Y-%m-%d")
    df["Time"] = pd.to_datetime(df["dtMsgStartTime2"]).dt.round("10min").dt.strftime("%H:%M")
    #df = df[(df["Time"] >= RELEVANT_TIME[0]) & (df["Time"] <= RELEVANT_TIME[1])]
    df["Toll"] = df["Zone_Toll"].astype(float)
    if df_all is None:
        df_all = df
    else:
        df_all = pd.concat([df_all, df], ignore_index = True)
df_all = df_all[["Time", "Toll"]].groupby(["Time"]).mean().reset_index().sort_values("Time")
df_all.to_csv("data/all_tolls.csv", index = False)