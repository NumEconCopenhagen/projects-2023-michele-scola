import numpy as np
from scipy import optimize

def u(L,alpha,kappa,nu,w,tau,G):
    """
    Q1
    return:
    utility function
    """
    C = kappa + (1 - tau)*w*L
    return np.log(C**alpha*(G**(1-alpha)))-nu*((L**2)/2)

def sol(alpha,kappa,nu,w,tau,G):
    """
    Q1
    return:
    optimal L (hours of labor)
    """
    obj = lambda x: -u(x,alpha,kappa,nu,w,tau,G)
    sol = optimize.minimize(obj,[12],method='SLSQP',bounds=[(0,24)],tol=1e-10)
    return sol.x

def utility(L,alpha,kappa,nu,w,tau):
    """
    Q3
    return:
    new utility function
    """
    C = kappa + (1-tau)*w*L
    G = tau * w * L * ((1-tau)*w)
    return np.log(C**alpha*(G**(1-alpha)))-nu*((L**2)/2)

def solve(alpha,kappa,nu,w,tau):
    """
    Q3
    return:
    optimal L (hours of labor)
    utility
    """
    obj = lambda x:  -utility(x,alpha,kappa,nu,w,tau)
    sol = optimize.minimize(obj,x0=8,method='SLSQP', bounds=[(0,24)],tol = 1e-10) 
    return sol.x, - sol.fun

def solve_tau(alpha,kappa,nu,w):
    """
    Q4
    return:
    optimal L
    optimal tau
    utility
    """
    obj = lambda x: -utility(x[0],alpha,kappa,nu,w,x[1])
    sol = optimize.minimize(obj,x0=(8,0.5),method='SLSQP', bounds=[(0,24),(0,1)],tol = 1e-10) 
    return sol.x[0], sol.x[1], sol.fun

def utility2(L,alpha,kappa,nu,w,tau,sigma,p,eps):
    """
    Q5-6
    return:
    new utility function
    """
    C = kappa + (1-tau)*w*L
    G = tau*w*L*((1-tau)*w)
    return ((((alpha*(C**((sigma-1)/sigma)) + ((1-alpha)*(G**((sigma-1)/sigma))))**sigma)**(1-p) - 1)/(1-p)) - nu*(L**(1+eps)/(1+eps))

def solve2(alpha,kappa,nu,w,tau,sigma,p,eps):
    """
    Q5
    return:
    optimal L
    """
    obj = lambda x: -utility2(x[0],alpha,kappa,nu,w,tau,sigma,p,eps)
    sol = optimize.minimize(obj,x0=(8),method='SLSQP', bounds=[(0,24)],tol = 1e-20) 
    return sol.x

def solve_tau2(alpha,kappa,nu,w,sigma,p,eps):
    """
    Q6
    return:
    optimal L
    optimal tau
    utility
    """
    obj = lambda x: -utility2(x[0],alpha,kappa,nu,w,x[1],sigma,p,eps)
    sol = optimize.minimize(obj,x0=(8,0.5),method='SLSQP', bounds=[(0,24),(0,1)],tol = 1e-10) 
    return sol.x[0], sol.x[1], sol.fun

def profit(x, par):
    """
    Q1
    return:
    profit function
    """
    return par.kappa*(x**(1-par.eta))-par.w*x

def solve3(par):
    """
    Q1
    return:
    optimal l (number of hairdressers)
    profit
    """
    obj = lambda x: -profit(x[0],par)
    sol = optimize.minimize(obj,x0=4,method='SLSQP',bounds=[(0,1000)],tol = 1e-20) 
    return sol.x, sol.fun

def lt(par):
    """
    Q1 (analytical solution)
    return:
    optimal l
    """
    return (((1-par.eta)*par.kappa)/par.w)**(1/par.eta)

def simulation(par):
    """
    Q2
    return:
    H (ex-ante expected value)
    """
    for k in range(par.K): #K simulations
        eps = np.random.normal(-0.5*(par.sigma_e**2), par.sigma_e, size = 120) #shock realization series
        for t in range(120): #calculating ex-post value for each period
            ex_post = np.zeros(120)
            par.kappa = par.rho*(np.log(k-1)) + eps[t] #update kappa
            par.kappa = np.exp(par.kappa)
            l_opt = lt(par) #calculating optimal level of hairdresser using lt() function
            profit = par.kappa*(l_opt**(1-par.eta)) - par.w*l_opt - 1*(l_opt!=l_opt)*par.iota #calulating profit
            ex_post_val = par.R**(-1) * profit #accumulating ex-post values
            ex_post[t] = ex_post_val #saving ex-post values
    return np.mean(ex_post) #calculating ex-ante expected value

def simulation2(par,delta):
    """
    Q3
    return:
    H (ex-ante expected value)
    """
    for k in range(par.K): #K simulations
        eps = np.random.normal(-0.5*(par.sigma_e**2), par.sigma_e, size = 120) #shock realization series
        for t in range(120): #calculating ex-post value for each period
            ex_post = np.zeros(120)
            par.kappa = par.rho*(np.log(k-1-1)) + eps[t-1] #previous kappa
            par.kappa = np.exp(par.kappa)
            L_ = lt(par) #previous optimal level of hairdresser
            par.kappa = par.rho*(np.log(k-1)) + eps[t] #update kappa
            par.kappa = np.exp(par.kappa)
            L_star = lt(par) #optimal level of hairdresser
            x = np.abs(L_ - L_star) #new policy
            if x > delta:
                l_opt = L_star
            else:
                l_opt = L_
            profit = par.kappa*(l_opt**(1-par.eta)) - par.w*l_opt - 1*(l_opt!=l_opt)*par.iota #profit
            ex_post_val = par.R**(-1) * profit #accumulating ex-post values
            ex_post[t] = ex_post_val #saving ex-post values
    return np.mean(ex_post) #calculating ex-ante expected value

def simulation3(par):
    """
    Q5
    return:
    H (ex-ante expected value)
    """
    l_opt = 0
    for k in range(par.K):
        eps = np.random.normal(-0.5*(par.sigma_e**2), par.sigma_e, size = 120)
        for t in range(120):
            ex_post = np.zeros(120)
            par.kappa = par.rho*(np.log(k-1)) + eps[t]
            par.kappa = np.exp(par.kappa)
            profit_ = par.kappa*(l_opt**(1-par.eta)) - par.w*l_opt - 1*(l_opt!=l_opt)*par.iota
            if profit_ > 0: #if profit > 0 --> increase hairdressers
                l_opt = lt(par)+(lt(par)*par.a)
            else: #if profit is negative --> decrease hairdressers
                l_opt = lt(par)+(lt(par)-par.b)
            profit = par.kappa*(l_opt**(1-par.eta)) - par.w*l_opt - 1*(l_opt!=l_opt)*par.iota
            ex_post_val = par.R**(-1) * profit
            ex_post[t] = ex_post_val
    return np.mean(ex_post)