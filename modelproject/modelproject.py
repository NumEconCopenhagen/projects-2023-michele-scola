from scipy import optimize
import numpy as np

#Stackelberg duopoly model functions

def leader_obj(x,a,b,c): #leader profit function
    x_follow = follower_opt(x[0],a,b,c)
    return -(a - (b*x[0]+b*x_follow) - c)*x[0]

def follower_obj(x,a,b,c): #follower profit function
    return -(a - (b*x[0]+b*x[1]) - c)*x[1]

def follower_opt(leader_prod,a,b,c): #follower optimization
    def obj(x):
        return follower_obj([leader_prod,x[0]],a,b,c)
    sol = optimize.minimize(obj,[5],bounds=[(0,10000)], method='SLSQP',tol = 1e-10)
    return sol.x[0]

def leader_opt(a,b,c): #leader optimization
    def obj(x):
        return leader_obj(x,a,b,c)
    sol = optimize.minimize(obj,[25], method='SLSQP', bounds=[(0,10000)],tol = 1e-10)
    return sol.x, follower_opt(sol.x,a,b,c)

def profits1_opt(a,b,c): #leader optimal profit
    sol_leader, sol_follower = leader_opt(a,b,c)
    p = a - (b*sol_leader+b*sol_follower)
    return (p - c)*sol_leader

def profits2_opt(a,b,c): #follower optimal profit
    sol_leader, sol_follower = leader_opt(a,b,c)
    p = a - (b*sol_leader+b*sol_follower)
    return (p - c)*sol_follower

#Stackelber model with 3 firms functions (extension)

def leader_obj_(x,a,b,c): #leader profit function
    x_follow1 = follower1_opt_(x[0],a,b,c)
    x_follow2 = follower2_opt_(x[0],x[1],a,b,c)
    return -(a - (b*x[0] + b*x_follow1 + b*x_follow2) - c)*x[0]

def follower1_obj_(x,a,b,c): #follower 1 profit function
    x_follow2 = follower2_opt_(x[0],x[1],a,b,c)
    return -(a - (b*x[0] + b*x[1] + b*x_follow2) - c)*x[1]

def follower2_obj_(x,a,b,c): #follower 2 profit function
    return -(a - (b*x[0] + b*x[1] + b*x[2]) - c)*x[2]

def follower2_opt_(leader_prod, follower1_prod, a,b,c): #follower 2 optimization 
    def obj(x):
        return follower2_obj_([leader_prod,follower1_prod,x[0]],a,b,c)
    sol = optimize.minimize(obj,[5],bounds=[(0,10000)], method='SLSQP',tol = 1e-13)
    return sol.x[0]

def follower1_opt_(leader_prod,a,b,c): #follower 1 optimization
    def obj(x):
        return follower1_obj_([leader_prod,x[0]],a,b,c)
    sol = optimize.minimize(obj,[5],bounds=[(0,10000)], method='SLSQP',tol = 1e-13)
    return sol.x[0]

def leader_opt_(a,b,c): #leader optimization
    def obj(x):
        return leader_obj_(x,a,b,c)
    sol = optimize.minimize(obj,[25,5], method='SLSQP', bounds=[(0,10000)],tol = 1e-13)
    return sol.x[0], follower1_opt_(sol.x[0],a,b,c), follower2_opt_(sol.x[0],sol.x[1],a,b,c)

