import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

#TAU_BAR = 15
GAMMA_BAR = 8 #10
BETA_BAR = 1.5 #300
A = 3
D = 10
CAPACITY = 50
## One set of working parameters: GAMMA_BAR = 1, BETA_BAR = 1, A = 3, D = 10, MU = 50, rho = 0.8, tau = 0.1

def cost(flow, rho):
    r = flow / (CAPACITY * rho)
    a = 0.15
    b = 0.4
    return 1 * (1 + a * (flow / CAPACITY) ** b) #r / (CAPACITY * rho - flow)

def c_o(x3, rho):
    flow = x3 * D
    return cost(flow, rho)

def c_h(x1, x2, rho):
    flow = (x1 + x2 / A) * D
    return cost(flow, rho)

def get_regime_x(x, tau, regime_type):
    if regime_type == "B":
        x1 = x
        x2 = tau / 2 * (1 / (GAMMA_BAR - tau) * x1 + 1 / GAMMA_BAR)
        x3 = 1 - x1 - x2
    else:
        x1 = 0
        x2 = x
        x3 = 1 - x2
    return x1, x2, x3

def regime_fixedpoint_loss(x, rho, tau, regime_type):
    x1, x2, x3 = get_regime_x(x, tau, regime_type)
    c_delta = c_o(x3, 1 - rho) - c_h(x1, x2, rho)
    if regime_type == "B":
        lhs = (1 - GAMMA_BAR / (GAMMA_BAR - tau) * x1) * c_delta
        rhs = tau / BETA_BAR
    elif regime_type == "A1":
        lhs = 1 / 2 * BETA_BAR / GAMMA_BAR * c_delta
        rhs = x2
    else:
        lhs = 1 / 2 * GAMMA_BAR / BETA_BAR / c_delta
        rhs = 1 - x2
    return (lhs - rhs) ** 2

def get_feasible_regime(rho, tau):
    first = GAMMA_BAR
    c_delta = c_o(1 - tau / (2 * GAMMA_BAR), 1 - rho) - c_h(0, tau / (2 * GAMMA_BAR), rho)
    sec = BETA_BAR * c_delta
    if tau < min(first, sec):
        ret = "B"
    elif tau > sec:
        ret = "A1"
    else:
        ret = "A2"
    return ret

def solve_regime(rho, tau, max_itr = 1000, eta = 0.1, eps = 1e-7, decay_sched = 500, eta_min = 1e-4):
    x = torch.tensor(0.5, requires_grad = True)
    regime_type = get_feasible_regime(rho, tau)
    loss_arr = []
    lb = torch.tensor(tau / (2 * GAMMA_BAR))
    ub = torch.tensor(1.)
    itr = 0
    loss = 1
    while itr < max_itr and loss > eps and eta >= eta_min:
        loss = regime_fixedpoint_loss(x, rho, tau, regime_type)
        loss_arr.append(float(loss.data))
        if torch.isnan(loss):
            eta = eta * 0.1
            x = torch.tensor(0.5, requires_grad = True)
            loss = 1
            itr = 0
        else:
            loss.backward()
            x.data = x.data - eta * x.grad
            x.grad.zero_()
#        x1.data = torch.min(torch.max(x1.data, lb), ub)
        if itr % decay_sched == 0 and itr > 0:
            eta = eta * 0.1
        itr += 1
    return get_regime_x(float(x.detach()), tau, regime_type), loss_arr

def get_congestion(x1, x2, x3, rho):
    return x3 * c_o(x3, 1 - rho) + (x1 + x2) * c_h(x1, x2, rho)

def get_revenue(x1, tau):
    return x1 * tau

rho_lst = [1/3, 2/3] #[0.1, 0.3, 0.5, 0.7, 0.8]
tau_lst = np.arange(1, 16) #np.linspace(0, TAU_BAR, num = 6)[1:]
arg_lst = list(itertools.product(rho_lst, tau_lst))
dct = {"rho": [], "tau": [], "x1": [], "x2": [], "x3": [], "loss": [], "congestion": [], "revenue": []}
for (rho, tau) in tqdm(arg_lst):
    (x1, x2, x3), loss_arr = solve_regime(rho, tau, max_itr = 2000, eta = 1e1, eps = 1e-7, decay_sched = 500, eta_min = 1e-4)
    congestion = get_congestion(x1, x2, x3, rho)
    revenue = get_revenue(x1, tau)
    dct["rho"].append(rho)
    dct["tau"].append(tau)
    dct["x1"].append(x1)
    dct["x2"].append(x2)
    dct["x3"].append(x3)
    dct["loss"].append(loss_arr[-1])
    dct["congestion"].append(congestion)
    dct["revenue"].append(revenue)
df = pd.DataFrame.from_dict(dct)
df.to_csv("results.csv", index = False)

#(x1, x2, x3), loss_arr = solve_regime(0.4, 3, max_itr = 2000, eta = 1e-1, eps = 1e-7, decay_sched = 500, eta_min = 1e-4)
#plt.plot(loss_arr)
#plt.xlabel("Iterations")
#plt.ylabel("Loss")
#plt.title(f"Loss = {loss_arr[-1]:.2e}\nx1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}")
#plt.savefig("loss.png")
#plt.clf()
#plt.close()
