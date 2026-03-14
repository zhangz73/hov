import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("toll_design_multiseg_hour=17_multi-rho.csv")
df.columns = ["HOT Capacity" if x == "Rho" else x for x in df.columns]
N_POP = 24546
df["Total Travel Time"] /= N_POP
df["Total Emission"] /= N_POP
df["Total Utility Cost"] /= N_POP

def compute_pareto_front(df, colname, xlabel, fname):
    df = df.sort_values("Total Revenue", ascending = True).copy()
    pareto = []
    pareto_sep = []
    df["pareto"] = 1
    df["pareto_sep"] = 1
    df["rank"] = df[colname].rank()
    for i in tqdm(range(df.shape[0])):
        is_pareto = 1
        is_pareto_sep = 1
        feat_i = df.iloc[i][colname]
        revenue_i = df.iloc[i]["Total Revenue"]
        rho_i = df.iloc[i]["HOT Capacity"]
        if df[(df[colname] < feat_i) & (df["Total Revenue"] > revenue_i)].shape[0] > 0:
            is_pareto = 0
        if df[(df[colname] < feat_i) & (df["Total Revenue"] > revenue_i) & (df["HOT Capacity"] == rho_i)].shape[0] > 0:
            is_pareto_sep = 0
#        for j in range(i + 1, df.shape[0]):
#            feat_j = df.iloc[j][colname]
#            revenue_j = df.iloc[j]["Total Revenue"]
#            rho_j = df.iloc[j]["HOT Capacity"]
#            if feat_j < feat_i and revenue_j > revenue_i:
#                is_pareto = 0
#                if rho_i == rho_j:
#                    is_pareto_sep = 0
        pareto.append(is_pareto)
        pareto_sep.append(is_pareto_sep)
    df["pareto"] = pareto
    df["pareto_sep"] = pareto_sep
    
    df_pareto = df[df["pareto"] == 1]

    for rho in [0.25, 0.5, 0.75]:
        df_sub = df[(df["HOT Capacity"] == rho) & (df["pareto_sep"] == 1)]
        plt.scatter(df_sub[colname], df_sub["Total Revenue"], label = f"$\\rho = {rho}$")
        plt.plot(df_sub[colname], df_sub["Total Revenue"])
#    plt.plot(df_pareto[colname], df_pareto["Total Revenue"], color = "black", alpha = 0.5, label = "Combined", linestyle = "--")
    plt.xlabel(xlabel)
    plt.ylabel("Total Revenue Gathered From Tolls ($)")
    #plt.title("Pareto Front")
    plt.legend()
    plt.savefig(f"DynamicDesign/MultiSeg/SingleHour/pareto_{fname}_combo.png")
    plt.clf()
    plt.close()

compute_pareto_front(df, "Total Travel Time", "Average Traffic Time Per Traveler (Minutes)", "latency")
compute_pareto_front(df, "Total Emission", "Average Emission Per Traveler (Minutes)", "emission")
compute_pareto_front(df, "Total Utility Cost", "Average Utility Cost Per Traveler ($)", "utility")
