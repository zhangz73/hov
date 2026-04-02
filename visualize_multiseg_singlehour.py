import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("toll_design_multiseg_hour=16_multi-rho.csv")
df.columns = ["HOT Capacity" if x == "Rho" else x for x in df.columns]
SEGMENT_NAMES = ["Auto Mall", "Mowry", "Decoto", "Whipple", "Hesperian"]
TOLL_COLS = [f"Toll {i}" for i in range(5)]
TOLL_VALS = [1.1, 2.2, 2.5, 4.0, 5.0]
TOLL_LAM = 0.001
#for toll_col in TOLL_COLS:
#    df = df[df[toll_col] > 0]
#toll_deviate = 0
#for i in range(len(SEGMENT_NAMES)):
#    toll_avg = TOLL_VALS[i]
#    toll_deviate += (toll_avg - df[f"Toll {i}"]).abs()
#df["Total Travel Time"] += toll_deviate * df["Total Travel Time"].mean() * TOLL_LAM
#df["Total Emission"] += toll_deviate * df["Total Emission"].mean() * TOLL_LAM
#df["Total Utility Cost"] += toll_deviate * df["Total Utility Cost"].mean() * TOLL_LAM
#df["Total Revenue"] -= toll_deviate * df["Total Revenue"].mean() * TOLL_LAM

df_demand = pd.read_csv("data/od_demand.csv")
N_POP = df_demand[df_demand["Hour"] == 16]["Demand"].sum() #1 #24546
df["Total Travel Time"] /= N_POP
df["Total Emission"] /= N_POP
df["Total Utility Cost"] /= N_POP

def compute_pareto_front(df, colname, xlabel, fname):
    df = df.sort_values("Total Revenue", ascending=True).copy()
    df["rank"] = df[colname].rank()

    # ------------------------------------------------------------------
    # Combined Pareto front:
    # A row is dominated iff min(colname) among rows with strictly larger
    # revenue is < current colname.
    # ------------------------------------------------------------------
    rev_min = (
        df.groupby("Total Revenue", as_index=False)[colname]
        .min()
        .sort_values("Total Revenue", ascending=False)
        .reset_index(drop=True)
    )

    # min feature among strictly higher revenue groups
    rev_min["min_feat_higher_rev"] = rev_min[colname].cummin().shift(1)
    df = df.merge(
        rev_min[["Total Revenue", "min_feat_higher_rev"]],
        on="Total Revenue",
        how="left",
    )

    df["pareto"] = (
        df["min_feat_higher_rev"].isna()
        | (df["min_feat_higher_rev"] >= df[colname])
    ).astype(int)

    # ------------------------------------------------------------------
    # Separate Pareto front within each HOT Capacity
    # ------------------------------------------------------------------
    rev_min_sep = (
        df.groupby(["HOT Capacity", "Total Revenue"], as_index=False)[colname]
        .min()
        .sort_values(["HOT Capacity", "Total Revenue"], ascending=[True, False])
        .reset_index(drop=True)
    )

    rev_min_sep["min_feat_higher_rev_sep"] = (
        rev_min_sep.groupby("HOT Capacity")[colname]
        .cummin()
        .groupby(rev_min_sep["HOT Capacity"])
        .shift(1)
    )

    df = df.merge(
        rev_min_sep[["HOT Capacity", "Total Revenue", "min_feat_higher_rev_sep"]],
        on=["HOT Capacity", "Total Revenue"],
        how="left",
    )

    df["pareto_sep"] = (
        df["min_feat_higher_rev_sep"].isna()
        | (df["min_feat_higher_rev_sep"] >= df[colname])
    ).astype(int)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    for rho in [0.25, 0.5]:
        df_sub = df[(df["HOT Capacity"] == rho) & (df["pareto_sep"] == 1)].copy()
        df_sub = df_sub.sort_values(colname, ascending=True)
        plt.scatter(df_sub[colname], df_sub["Total Revenue"], label=f"$\\rho = {rho}$")
        plt.plot(df_sub[colname], df_sub["Total Revenue"])

    plt.xlabel(xlabel)
    plt.ylabel("Total Revenue Gathered From Tolls ($)")
    plt.legend()
    plt.savefig(f"DynamicDesign/MultiSeg/SingleHour/pareto_{fname}_combo.png")
    plt.clf()
    plt.close()

    # optional: clean helper columns before returning
    df = df.drop(columns=["min_feat_higher_rev", "min_feat_higher_rev_sep"])
    return df

def fmt_toll(x):
    return f"$\\${x:.1f}$"

def pick_best_row(df_sub, objective_col, minimize=True):
    if minimize:
        idx = df_sub[objective_col].idxmin()
    else:
        idx = df_sub[objective_col].idxmax()
    return df_sub.loc[idx]


def get_objective_column_map(df):
    """
    Adjust here if your dataframe uses different names.
    """
    col_map = {}

    if "Total Travel Time" in df.columns:
        col_map["Agent Time Minimization"] = ("Total Travel Time", True)
    elif "Total Agent Time" in df.columns:
        col_map["Agent Time Minimization"] = ("Total Agent Time", True)
    else:
        raise ValueError("Cannot find a column for Agent Time Minimization.")

    if "Total Emission" in df.columns:
        col_map["Vehicle Time Minimization"] = ("Total Emission", True)
    elif "Total Vehicle Time" in df.columns:
        col_map["Vehicle Time Minimization"] = ("Total Vehicle Time", True)
    else:
        raise ValueError("Cannot find a column for Vehicle Time Minimization.")

    if "Total Revenue" in df.columns:
        col_map["Revenue Maximization"] = ("Total Revenue", False)
    else:
        raise ValueError("Cannot find column 'Total Revenue'.")

    if "Total Utility Cost" in df.columns:
        col_map["Cost Minimization"] = ("Total Utility Cost", True)
    elif "Total Cost" in df.columns:
        col_map["Cost Minimization"] = ("Total Cost", True)
    else:
        raise ValueError("Cannot find a column for Cost Minimization.")

    return col_map

def make_latex_table(df, rho_values=(0.25, 0.50), output_path=None):
    df = df.copy()
    obj_map = get_objective_column_map(df)

    lines = []
    lines.append("\\begin{tabular}{|c|c|c|c|c|c|c|}")
    lines.append("        \\hline")
    lines.append("        \\multirow{2}{*}{HOT Capacity} & \\multirow{2}{*}{Objective} & \\multicolumn{5}{c|}{Toll Prices} \\\\")
    lines.append("         & & Auto Mall & Mowry & Decoto & Whipple & Hesperian\\\\")
    lines.append("        \\hline")

    for rho_idx, rho in enumerate(rho_values):
        df_rho = df[df["HOT Capacity"] == rho].copy()
        if len(df_rho) == 0:
            continue

        # Objective rows
        obj_rows = []
        for obj_name, (obj_col, minimize) in obj_map.items():
            row = pick_best_row(df_rho, obj_col, minimize=minimize)
            toll_str = " & ".join(fmt_toll(row[c]) for c in TOLL_COLS)
            obj_rows.append((obj_name, toll_str))

        if len(obj_rows) > 0:
            first_obj, first_tolls = obj_rows[0]
            lines.append(f"        \\multirow{{{len(obj_rows)}}}{{*}}{{{rho:.2f}}} & {first_obj} & {first_tolls}\\\\")
            for obj_name, toll_str in obj_rows[1:]:
                lines.append(f"         & {obj_name} & {toll_str}\\\\")
            lines.append("        \\hline")

    lines.append("    \\end{tabular}")

    latex_str = "\n".join(lines)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(latex_str)

    return latex_str
compute_pareto_front(df, "Total Travel Time", "Average Traffic Time Per Traveler (Minutes)", "latency")
compute_pareto_front(df, "Total Emission", "Average Emission Per Traveler (Minutes)", "emission")
compute_pareto_front(df, "Total Utility Cost", "Average Utility Cost Per Traveler ($)", "utility")

latex_table = make_latex_table(df, rho_values=(0.25, 0.50, 0.75))
print(latex_table)
