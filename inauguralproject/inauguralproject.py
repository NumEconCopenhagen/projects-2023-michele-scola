from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        par.nu_men = 1

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.LM = np.nan
        sol.HM = np.nan
        sol.LF = np.nan
        sol.HF = np.nan

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        sol.alpha_sol = np.nan
        sol.sigma_sol = np.nan
        sol.nu_men = np.nan

        sol.log_ratio_vec = np.zeros(par.wF_vec.size)
        sol.log_wages_vec = np.zeros(par.wF_vec.size)

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            HM = np.fmax(HM, 1e-07)
            HF = np.fmax(HF, 1e-07)
            inside = (1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma)
            inside =np.fmax(inside, 1e-07)
            H = (inside)**(par.sigma/(par.sigma - 1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        sol.LM = LM[j]
        sol.HM = HM[j]
        sol.LF = LF[j]
        sol.HF = HF[j]

        # e. print
        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol

    def solve(self, do_print=False):
        """ solve model continously """
        
        par = self.par
        sol = self.sol

        #objective function
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])
        
        #constraints
        constraint1 = lambda x: 24 - (x[0] + x[1])
        constraint2 = lambda x: 24 - (x[2] + x[3])
        cons1 = ({'type':'ineq','fun':constraint1})
        cons2 = ({'type':'ineq','fun':constraint2})
        constraints = [cons1,cons2]
        
        #bounds
        bounds = ((0,24),(0,24),(0,24),(0,24))
        
        #initial guess
        x_guess = (5,5,5,5)

        #call solver
        solution = optimize.minimize(obj, x_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol = 1e-10)
        
        #save results
        sol.LM = solution.x[0]
        sol.HM = solution.x[1]
        sol.LF = solution.x[2]
        sol.HF = solution.x[3]

        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        #solve with contiuous choice set
        if discrete == False:
            for i,wf in enumerate(par.wF_vec):
                par.wF = wf
                self.solve()
                sol.HF_vec[i] = sol.HF
                sol.HM_vec[i] = sol.HM
                sol.LM_vec[i] = sol.LM
                sol.LF_vec[i] = sol.LF

        #solve with discrete choice set
        else:
            for i, wf in enumerate(par.wF_vec):
                par.wF = wf
                self.solve_discrete()
                sol.HF_vec[i] = sol.HF
                sol.HM_vec[i] = sol.HM
                sol.LM_vec[i] = sol.LM
                sol.LF_vec[i] = sol.LF
        return sol.HF_vec, sol.HM_vec, sol.LM_vec, sol.LF_vec


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        self.solve_wF_vec()

        x = np.log(par.wF_vec)
        y = np.log((sol.HF_vec/sol.HM_vec))
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return sol.beta0, sol.beta1

    
    def estimate(self):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol

        #define objective function
        def obj(x):
            par.alpha = x[0]
            par.sigma = x[1]
            self.run_regression()
            return (par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
        
        #initial guess
        initial_guess=(0.9, 0.15)

        #bounds
        bounds = ((0.01,0.99),(0.01,0.99))

        #minimizer
        res = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds, tol = 1e-08)

        #store solutiomns
        sol.alpha_sol = res.x[0]
        sol.sigma_sol = res.x[1]
        
        return print(f'alpha = {sol.alpha_sol}, sigma = {sol.sigma_sol}')
    
    def estimate_5(self, alpha_5):
        """ estimate sigma with alpha as an input """

        par = self.par
        sol = self.sol

        #define objective function
        def obj(x):
            par.alpha = alpha_5
            par.sigma = x[0]
            par.nu_men = x[1]
            self.run_regression_5()
            return (par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
        
        #initial guess
        initial_guess=[(0.3),(0.8)]

        #bounds
        bounds = [(0.01,0.99),(0.01,0.99)]

        #minimizer
        res = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds, tol = 1e-08)

        #store solutiomns
        sol.sigma_sol = res.x[0]
        sol.nu_men = res.x[1]
        
        return print(f'sigma = {sol.sigma_sol}, nu_men = {sol.nu_men}')
    
    def calc_utility_5(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            HM = np.fmax(HM, 1e-07)
            HF = np.fmax(HF, 1e-07)
            inside = (1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma)
            inside =np.fmax(inside, 1e-07)
            H = (inside)**(par.sigma/(par.sigma - 1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_) + par.nu_men*HM
        
        return utility - disutility
    
    def solve_5(self, do_print=False):
        """ solve model continously """
        
        par = self.par
        sol = self.sol

        #objective function
        obj = lambda x: -self.calc_utility_5(x[0],x[1],x[2],x[3])
        
        #constraints
        constraint1 = lambda x: 24 - (x[0] + x[1])
        constraint2 = lambda x: 24 - (x[2] + x[3])
        cons1 = ({'type':'ineq','fun':constraint1})
        cons2 = ({'type':'ineq','fun':constraint2})
        constraints = [cons1,cons2]
        
        #bounds
        bounds = ((0,24),(0,24),(0,24),(0,24))
        
        #initial guess
        x_guess = (5,5,5,5)

        #call solver
        solution = optimize.minimize(obj, x_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol = 1e-10)
        
        #save results
        sol.LM = solution.x[0]
        sol.HM = solution.x[1]
        sol.LF = solution.x[2]
        sol.HF = solution.x[3]

        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol
    
    def run_regression_5(self):
        """ run regression """

        par = self.par
        sol = self.sol

        for i,wf in enumerate(par.wF_vec):
                par.wF = wf
                self.solve_5()
                sol.HF_vec[i] = sol.HF
                sol.HM_vec[i] = sol.HM
                sol.LM_vec[i] = sol.LM
                sol.LF_vec[i] = sol.LF

        x = np.log(par.wF_vec)
        y = np.log((sol.HF_vec/sol.HM_vec))
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return sol.beta0, sol.beta1
