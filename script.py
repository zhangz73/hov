import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
import matplotlib.pyplot as plt
from tqdm import tqdm

#test 
#TAU_BAR = 15
UGAMMA = 8 #10
UBETA = 1.5 #300
LGAMMA = 0
LBETA = 0
A = 2.5
D = 115.13 #GAMMA_BAR * BETA_BAR
CAPACITY = 140
TIME = 22
INT_GRAN = 100
F_BETA_GAMMA_LST = None

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

def f(beta, gamma):
#    return 1 / (UBETA * UGAMMA)
    coef = 1 / (UBETA * UGAMMA)
    if int(gamma) % 2 == 0:
        coef *= 2
    else:
        coef = 0
    return coef

def populate_f_beta_gamma_lst():
    global F_BETA_GAMMA_LST
    if F_BETA_GAMMA_LST is None:
        f_beta_points = np.linspace(LBETA, UBETA, INT_GRAN + 1)
        f_beta_points = (f_beta_points[1:] + f_beta_points[:-1]) / 2
        f_gamma_points = np.linspace(LGAMMA, UGAMMA, INT_GRAN + 1)
        f_gamma_points = (f_gamma_points[1:] + f_gamma_points[:-1]) / 2
        F_BETA_GAMMA_LST = torch.zeros((INT_GRAN, INT_GRAN))
        for i in range(INT_GRAN):
            for j in range(INT_GRAN):
                beta = f_beta_points[i]
                gamma = f_gamma_points[j]
                F_BETA_GAMMA_LST[i, j] = f(beta, gamma)

def f_int(lbeta, ubeta, gamma_pos):
    global F_BETA_GAMMA_LST
    if F_BETA_GAMMA_LST is None:
        populate_f_beta_gamma_lst()
    beta_gap = (UBETA - LBETA) / INT_GRAN
    lbeta_pos = int(round(((float(lbeta) - LBETA) / (UBETA - LBETA)) * INT_GRAN))
    ubeta_pos = int(round(((float(ubeta) - LBETA) / (UBETA - LBETA)) * INT_GRAN))
    ans = torch.sum(F_BETA_GAMMA_LST[lbeta_pos:(ubeta_pos+1),gamma_pos] * beta_gap)
    return ans

def regime_fixedpoint_loss(x, rho, tau):
    # x: toll, pool, ordinary
    c_delta = c_o(x[2], rho) - c_h(x[0], x[1], rho)
    sigma_toll_rhs = 0
    gamma_gap = (UGAMMA - LGAMMA) / INT_GRAN
    gamma_lo_pos = min(int(((tau - LGAMMA) / (UGAMMA - LGAMMA)) * INT_GRAN), INT_GRAN - 1)
    for gamma_pos in range(gamma_lo_pos, INT_GRAN):
        sigma_toll_rhs += gamma_gap * f_int(tau / c_delta, UBETA, gamma_pos)
    sigma_pool_rhs = 0
    for gamma_pos in range(0, gamma_lo_pos):
        gamma = gamma_pos * gamma_gap + LGAMMA
        sigma_pool_rhs += gamma_gap * f_int(gamma / c_delta, UBETA, gamma_pos)
    return (x[0] - sigma_toll_rhs) ** 2 + (x[1] - sigma_pool_rhs) ** 2

def solve_regime(rho, tau, max_itr = 1000, eta = 0.1, eps = 1e-7, decay_sched = 500, eta_min = 1e-4):
    x_init = [1/3, 1/3, 1/3]
    x = torch.tensor(x_init, requires_grad = True)
    itr = 0
    loss = 1
    loss_arr = []
#    x = optimize.fixed_point(regime_fixedpoint, [0.5], args = (rho, tau, regime_type))[0]
    while itr < max_itr and loss > eps and eta >= eta_min:
        loss = regime_fixedpoint_loss(x, rho, tau)
        loss_arr.append(float(loss.data))
        if torch.isnan(loss):
            eta = eta * 0.1
            x = torch.tensor(x_init, requires_grad = True)
            loss = 1
            itr = 0
        else:
            loss.backward()
            eta_curr = eta
            x.data = x.data - eta_curr * x.grad
            x.grad.zero_()
#        x.data = torch.min(torch.max(x.data, lb), ub)
        if itr % decay_sched == 0 and itr > 0:
            eta = eta * 0.1
#            print(eta, float(loss.data), float(x.data))
            x = torch.tensor(x_init, requires_grad = True)
            itr = 0
        itr += 1
    loss = regime_fixedpoint_loss(x, rho, tau)
    loss_arr.append(float(loss.data))
    x = x.detach().numpy()
    x[2] = 1 - x[0] - x[1]
    return x, loss_arr

def get_congestion(x1, x2, x3, rho):
    return x3 * c_o(x3, 1 - rho) + (x1 + x2) * c_h(x1, x2, rho)

def get_revenue(x1, tau):
    return x1 * tau * D

rho_lst = [1/4, 2/4, 3/4] #[0.1, 0.3, 0.5, 0.7, 0.8]
tau_lst = np.arange(0, 10, 0.5)[1:] #np.linspace(0, TAU_BAR, num = 6)[1:]
arg_lst = list(itertools.product(rho_lst, tau_lst))
dct = {"rho": [], "tau": [], "x1": [], "x2": [], "x3": [], "loss": [], "congestion": [], "revenue": []}
for (rho, tau) in tqdm(arg_lst):
    x, loss_arr = solve_regime(rho, tau, max_itr = 1000, eta = 1e-1, eps = 1e-4, decay_sched = 500, eta_min = 1e-8)
    x1, x2, x3 = x[0], x[1], x[2]
    congestion = get_congestion(x1, x2, x3, rho)
    revenue = get_revenue(x1, tau)
    dct["rho"].append(rho)
    dct["tau"].append(tau)
    dct["x1"].append(x1)
    dct["x2"].append(x2)
    dct["x3"].append(x3)
    dct["loss"].append(loss_arr[-1])
#    dct["regime"].append(regime_type)
    dct["congestion"].append(congestion)
    dct["revenue"].append(revenue)
df = pd.DataFrame.from_dict(dct)
df.to_csv("results.csv", index = False)

#x, loss_arr = solve_regime(0.25, 1, max_itr = 1000, eta = 1e-1, eps = 1e-4, decay_sched = 2000, eta_min = 1e-7)
#print(x)
##print(loss_arr)
#plt.plot(loss_arr)
#plt.xlabel("Iterations")
#plt.ylabel("Loss")
#plt.title(f"Loss = {loss_arr[-1]:.2e}\nx1 = {x[0]:.2f}, x2 = {x[1]:.2f}, x3 = {x[2]:.2f}")
#plt.savefig("loss.png")
#plt.clf()
#plt.close()
