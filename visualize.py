import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
pareto = []
pareto_sep = []
for i in range(df.shape[0]):
    is_pareto = 1
    is_pareto_sep = 1
    congestion_i = df.iloc[i]["congestion"]
    revenue_i = df.iloc[i]["revenue"]
    rho_i = df.iloc[i]["rho"]
    for j in range(i + 1, df.shape[0]):
        congestion_j = df.iloc[j]["congestion"]
        revenue_j = df.iloc[j]["revenue"]
        rho_j = df.iloc[j]["rho"]
        if congestion_j < congestion_i and revenue_j > revenue_i:
            is_pareto = 0
            if rho_i == rho_j:
                is_pareto_sep = 0
            break
    pareto.append(is_pareto)
    pareto_sep.append(is_pareto_sep)
df["pareto"] = pareto
df["pareto_sep"] = pareto_sep

df_pareto = df[df["pareto"] == 1]
plt.scatter(df_pareto["congestion"], df_pareto["revenue"])
plt.xlabel("Congestion")
plt.ylabel("Revenue")
plt.title("Pareto Front")
plt.savefig("pareto.png")
plt.clf()
plt.close()

for rho in [0.25, 0.5, 0.75]:
    df_sub = df[(df["rho"] == rho) & (df["pareto_sep"] == 1)]
    plt.scatter(df_sub["congestion"], df_sub["revenue"], label = f"rho = {rho}")
plt.xlabel("Congestion")
plt.ylabel("Revenue")
plt.title("Pareto Front")
plt.legend()
plt.savefig("pareto_sep.png")
plt.clf()
plt.close()
