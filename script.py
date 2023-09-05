import itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy.stats import multivariate_normal
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
TIME = 22 #30 for non-convex shape
INT_GRAN = 100
F_BETA_GAMMA_LST = None

var = multivariate_normal(mean = [(UBETA + LBETA) / 2, (UGAMMA + LGAMMA) / 2], cov = [[(UBETA + LBETA) / 2, 0], [0, (UGAMMA + LGAMMA) / 2]])

def cost(flow, rho):
    r = flow / (CAPACITY * rho)
    a = 0.15
    b = 1#4
    return TIME * (1 + a * (flow / (CAPACITY * rho)) ** b)

def c_o(x3, rho):
    flow = x3 * D
    return cost(flow, rho)

def c_h(x1, x2, rho):
    flow = (x1 + x2 / A) * D
    return cost(flow, rho)

def f(beta, gamma):
    return 1 / (UBETA * UGAMMA)
#    coef = 1 / (UBETA * UGAMMA)
#    if int(gamma) % 4 == 1:
#        coef *= 4#16/7
#    else:
#        coef *= 0#4/7
#    return coef
#    return var.pdf([beta, gamma])

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
    lbeta_pos = min(lbeta_pos, INT_GRAN - 1)
    ubeta_pos = min(ubeta_pos, INT_GRAN - 1)
    ans = torch.sum(F_BETA_GAMMA_LST[lbeta_pos:(ubeta_pos+1),gamma_pos] * beta_gap)
    return ans

def f_int_beta(lgamma, ugamma, beta_pos):
    global F_BETA_GAMMA_LST
    if F_BETA_GAMMA_LST is None:
        populate_f_beta_gamma_lst()
    gamma_gap = (UGAMMA - LGAMMA) / INT_GRAN
    lgamma_pos = int(round(((float(lgamma) - LGAMMA) / (UGAMMA - LGAMMA)) * INT_GRAN))
    ugamma_pos = int(round(((float(ugamma) - LGAMMA) / (UGAMMA - LGAMMA)) * INT_GRAN))
    lgamma_pos = min(lgamma_pos, INT_GRAN - 1)
    ugamma_pos = min(ugamma_pos, INT_GRAN - 1)
    ans = torch.sum(F_BETA_GAMMA_LST[beta_pos, lgamma_pos:(ugamma_pos+1)] * gamma_gap)
    return ans

def regime_fixedpoint_loss(x, rho, tau):
    # x: toll, pool, ordinary
    c_delta = c_o(1 - x[0] - x[1], 1 - rho) - c_h(x[0], x[1], rho)
    sigma_toll_rhs = 0
    gamma_gap = (UGAMMA - LGAMMA) / INT_GRAN
    gamma_lo_pos = min(int(((tau - LGAMMA) / (UGAMMA - LGAMMA)) * INT_GRAN), INT_GRAN - 1)
    for gamma_pos in range(gamma_lo_pos, INT_GRAN):
        sigma_toll_rhs += gamma_gap * f_int(tau / c_delta, UBETA, gamma_pos)
    sigma_pool_rhs = 0
    for gamma_pos in range(0, gamma_lo_pos + 1):
        gamma = gamma_pos * gamma_gap + LGAMMA
        sigma_pool_rhs += gamma_gap * f_int(gamma / c_delta, UBETA, gamma_pos)
    return (x[0] - sigma_toll_rhs) ** 2 + (x[1] - sigma_pool_rhs) ** 2

def solve_regime(rho, tau, max_itr = 1000, eta = 0.1, eps = 1e-7, decay_sched = 500, eta_min = 1e-4):
    pool_cutoff = 0
    beta_gap = (UBETA - LBETA) / INT_GRAN
    for beta_pos in range(INT_GRAN):
        beta = beta_pos * beta_gap + LBETA
        pool_cutoff += beta_gap * f_int_beta(0, tau * beta / UBETA, beta_pos)
    pool_cutoff = float(pool_cutoff)
    c_delta_cutoff = c_o(1 - pool_cutoff, 1 - rho) - c_h(0, pool_cutoff, rho)
    if tau < min(UGAMMA, UBETA * c_delta_cutoff):
        x_init = [pool_cutoff / 2, pool_cutoff / 2]
        regime_type = "B"
    else:
        if tau > UBETA * c_delta_cutoff:
            regime_type = "A1"
            x_init = [0, pool_cutoff / 2]
        else:
            regime_type = "A2"
            x_init = [0, (1 - pool_cutoff) / 2]
    x = torch.tensor(x_init, requires_grad = True)
    itr = 0
    loss = 1
    loss_arr = []
#    x = optimize.fixed_point(regime_fixedpoint, [0.5], args = (rho, tau, regime_type))[0]
    while itr < max_itr and loss > eps and eta >= eta_min:
        loss = regime_fixedpoint_loss(x, rho, tau)
        loss_arr.append(float(loss.data))
        if loss <= eps:
            break
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
    x1, x2 = x[0], x[1]
    x3 = 1 - x1 - x2
    return (x1, x2, x3), regime_type, loss_arr

def get_congestion(x1, x2, x3, rho):
    return x3 * c_o(x3, 1 - rho) + (x1 + x2) * c_h(x1, x2, rho)
#    return  x3 * c_o(x3, 1 - rho) + (x1 + x2 / A) * c_h(x1, x2, rho)

def get_revenue(x1, tau):
    return x1 * tau * D

rho_lst = [1/4, 2/4, 3/4] #[0.1, 0.3, 0.5, 0.7, 0.8]
tau_lst = np.arange(0, 10, 0.2)[1:] #np.linspace(0, TAU_BAR, num = 6)[1:]
arg_lst = list(itertools.product(rho_lst, tau_lst))
dct = {"rho": [], "tau": [], "x1": [], "x2": [], "x3": [], "loss": [], "regime": [], "congestion": [], "revenue": []}
for (rho, tau) in tqdm(arg_lst):
    (x1, x2, x3), regime_type, loss_arr = solve_regime(rho, tau, max_itr = 2000, eta = 1e-2, eps = 1e-6, decay_sched = 2000, eta_min = 1e-8)
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

#loss = regime_fixedpoint_loss(torch.tensor([0.174, 0.075, 0.751]), 0.25, 1) #[0.287, 0.03, 0.683]
#print(loss)

#(x1, x2, x3), regime_type, loss_arr = solve_regime(0.25, 1, max_itr = 1000, eta = 1e-3, eps = 1e-5, decay_sched = 2000, eta_min = 1e-7)
#print(x1, x2, x3)
##print(loss_arr)
#plt.plot(loss_arr)
#plt.xlabel("Iterations")
#plt.ylabel("Loss")
#plt.title(f"Loss = {loss_arr[-1]:.2e}\nx1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}")
#plt.savefig("loss.png")
#plt.clf()
#plt.close()
