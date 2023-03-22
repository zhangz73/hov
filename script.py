import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
import matplotlib.pyplot as plt
from tqdm import tqdm

#TAU_BAR = 15
GAMMA_BAR = 8 #10
BETA_BAR = 1.5 #300
A = 2.5
D = 115.13 #GAMMA_BAR * BETA_BAR
CAPACITY = 140
TIME = 22
## One set of working parameters: GAMMA_BAR = 1, BETA_BAR = 1, A = 3, D = 10, MU = 50, rho = 0.8, tau = 0.1

def cost(flow, rho):
    r = flow / (CAPACITY * rho)
    a = 0.15
    b = 4
    return TIME * (1 + a * (flow / (CAPACITY * rho)) ** b) #r / (CAPACITY * rho - flow)

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
        lhs = 1 / 2 * GAMMA_BAR / BETA_BAR
        rhs = (1 - x2) * c_delta
    if regime_type in ["B", "A1", "A2"]:
        loss = ((lhs - rhs) / TIME) ** 2
    else:
        loss = ((lhs - rhs) * TIME) ** 2
#    loss = (lhs - rhs) ** 2
#    print(lhs, rhs, loss)
    return loss

def regime_fixedpoint(x, rho, tau, regime_type):
    x1, x2, x3 = get_regime_x(x, tau, regime_type)
    c_delta = c_o(x3, 1 - rho) - c_h(x1, x2, rho)
    if regime_type == "B":
        lhs = (1 - tau / (BETA_BAR * c_delta)) / (GAMMA_BAR / (GAMMA_BAR - tau))
    elif regime_type == "A1":
        lhs = 1 / 2 * BETA_BAR / GAMMA_BAR * c_delta
    else:
        lhs = 1 - 1 / 2 * GAMMA_BAR / BETA_BAR / c_delta
    return lhs

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

def cost_is_feasible(rho, tau):
    c_delta = c_o(0, 1 - rho) - c_h(1 - tau / GAMMA_BAR, tau / GAMMA_BAR, rho)
    return c_delta < 0

def solve_regime(rho, tau, max_itr = 1000, eta = 0.1, eps = 1e-7, decay_sched = 500, eta_min = 1e-4):
    assert cost_is_feasible(rho, tau)
    regime_type = get_feasible_regime(rho, tau)
    loss_arr = []
    if regime_type == "B":
        lb = torch.tensor(0.)
        ub = torch.tensor(1 - tau / GAMMA_BAR)
        x_init = (1 - tau / GAMMA_BAR) / 2
    elif regime_type == "A1":
        lb = torch.tensor(0.)
        ub = torch.tensor(tau / (2 * GAMMA_BAR))
        x_init = tau / (2 * GAMMA_BAR) / 2
    else:
        lb = torch.tensor(tau / (2 * GAMMA_BAR))
        ub = torch.tensor(1.)
        x_init = (1 + tau / (2 * GAMMA_BAR)) / 2
    x = torch.tensor(x_init, requires_grad = True)
    itr = 0
    loss = 1
#    x = optimize.fixed_point(regime_fixedpoint, [0.5], args = (rho, tau, regime_type))[0]
    while itr < max_itr and loss > eps and eta >= eta_min:
        loss = regime_fixedpoint_loss(x, rho, tau, regime_type)
        loss_arr.append(float(loss.data))
        if torch.isnan(loss):
            eta = eta * 0.1
            x = torch.tensor(x_init, requires_grad = True)
            loss = 1
            itr = 0
        else:
            loss.backward()
            eta_curr = eta
            tmp = x.data - eta_curr * x.grad
            while (tmp > ub or tmp < lb) and eta_curr >= eta_min:
                eta_curr = eta_curr * 0.1
                tmp = x.data - eta_curr * x.grad
            if tmp >= lb and tmp <= ub:
                x.data = tmp
            x.grad.zero_()
#        x.data = torch.min(torch.max(x.data, lb), ub)
        if itr % decay_sched == 0 and itr > 0:
            eta = eta * 0.1
#            print(eta, float(loss.data), float(x.data))
            x = torch.tensor(x_init, requires_grad = True)
            itr = 0
        itr += 1
    x = float(x.detach())
    loss = regime_fixedpoint_loss(x, rho, tau, regime_type)
    loss_arr.append(loss)
    return get_regime_x(x, tau, regime_type), regime_type, loss_arr

def get_congestion(x1, x2, x3, rho):
    return x3 * c_o(x3, 1 - rho) + (x1 + x2) * c_h(x1, x2, rho)

def get_revenue(x1, tau):
    return x1 * tau * D

rho_lst = [1/4, 2/4, 3/4] #[0.1, 0.3, 0.5, 0.7, 0.8]
tau_lst = np.arange(0, 10, 0.5)[1:] #np.linspace(0, TAU_BAR, num = 6)[1:]
arg_lst = list(itertools.product(rho_lst, tau_lst))
dct = {"rho": [], "tau": [], "x1": [], "x2": [], "x3": [], "loss": [], "regime": [], "congestion": [], "revenue": []}
for (rho, tau) in tqdm(arg_lst):
    (x1, x2, x3), regime_type, loss_arr = solve_regime(rho, tau, max_itr = 10000, eta = 1e-1, eps = 1e-7, decay_sched = 2000, eta_min = 1e-8)
    congestion = get_congestion(x1, x2, x3, rho)
    revenue = get_revenue(x1, tau)
    dct["rho"].append(rho)
    dct["tau"].append(tau)
    dct["x1"].append(x1)
    dct["x2"].append(x2)
    dct["x3"].append(x3)
    dct["loss"].append(loss_arr[-1])
    dct["regime"].append(regime_type)
    dct["congestion"].append(congestion)
    dct["revenue"].append(revenue)
df = pd.DataFrame.from_dict(dct)
df.to_csv("results.csv", index = False)

#(x1, x2, x3), regime_type, loss_arr = solve_regime(0.75, 7.5, max_itr = 10000, eta = 1e-1, eps = 1e-7, decay_sched = 2000, eta_min = 1e-7)
#print(x1, x2, x3)
##print(loss_arr)
#plt.plot(loss_arr)
#plt.xlabel("Iterations")
#plt.ylabel("Loss")
#plt.title(f"Loss = {loss_arr[-1]:.2e}\nx1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}")
#plt.savefig("loss.png")
#plt.clf()
#plt.close()
