from parameters import Parameters
from model import initial_values, kinetics
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np



def singleRun():
    par = Parameters()
    init = initial_values()
    # print(init)
    tspan=(0,50)
    sol = solve_ivp(kinetics, t_span=tspan, y0=init, args=(par,))
    plot(sol, par)

def plot(sol, par):
    plt.figure(1)
    plt.plot(sol.t, sol.y[0,:], label='Glucose')
    plt.plot(sol.t, sol.y[1,:], label='Xylose')
    plt.plot(sol.t, sol.y[2,:], label='Furfural')
    plt.plot(sol.t, sol.y[3,:], label='Furfuryl alcohol')
    plt.plot(sol.t, sol.y[4,:], label='5-HMF')
    plt.plot(sol.t, sol.y[5,:], label='HAc')
    plt.plot(sol.t, sol.y[6,:], label='Ethanol')
    plt.plot(sol.t, sol.y[7,:], label='Biomass')
    plt.xlim(0,50)
    plt.ylim(0,40)
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration [g/L]')
    plt.legend()
    plt.savefig('growth.png')



def localSensitivityAnalysis():

    DISTURBANCE = 1.1
    tol = 1e-10
    method = 'LSODA'

    par = Parameters()
    init = initial_values()
    tspan=(0,50)
    time = np.arange(0,50,0.1)
    sol = solve_ivp(kinetics, t_span=tspan, y0=init, args=(par,), t_eval=time, method=method,  atol=tol, rtol=tol)
    par.numaxG_orig = 3.9872         # max consumption rate of glucose (1.0005)  h^-1       [1]
    par.KSPG_orig   = 0.14637        # affinity constant glucose to ethanol      g/L        [1]
    par.KiPG_orig   = 4752.8         # inhibition constant glucose to ethanol    g/L        [1]

    # PAR 1
    par.numaxG = par.numaxG_orig*DISTURBANCE
    par.KSPG = par.KSPG_orig 
    par.KiPG = par.KiPG_orig 
    sol1 = solve_ivp(kinetics, t_span=tspan, y0=init, args=(par,), t_eval=time, method=method,atol=tol, rtol=tol)
    dy1dp = (sol1.y-sol.y)/(par.numaxG-par.numaxG_orig) 
    

    # PAR 1
    par.numaxG = par.numaxG_orig 
    par.KSPG = par.KSPG_orig * DISTURBANCE
    par.KiPG = par.KiPG_orig 
    sol2 = solve_ivp(kinetics, t_span=tspan, y0=init, args=(par,), t_eval=time,  method=method,atol=tol, rtol=tol)
    dy2dp = (sol2.y-sol.y)/(par.KSPG-par.KSPG_orig) 


    # PAR 1
    par.numaxG = par.numaxG_orig
    par.KSPG = par.KSPG_orig 
    par.KiPG = par.KiPG_orig * DISTURBANCE
    sol3 = solve_ivp(kinetics, t_span=tspan, y0=init, args=(par,), t_eval=time,  method=method,atol=tol, rtol=tol)
    dy3dp = (sol3.y-sol.y)/(par.KiPG-par.KiPG_orig) 



    fig, axs = plt.subplots(nrows=3, ncols=8, sharex=True, squeeze=True, constrained_layout=True)
    i = 0
    for n in par.var_names:
        axs[0,i].plot(sol1.t, dy1dp[i,:])
        axs[0,i].set_title(n)
        i+=1
    for i in range(8):
        axs[1,i].plot(sol1.t, dy2dp[i,:])
    for i in range(8):
        axs[2,i].plot(sol1.t, dy3dp[i,:])
    axs[0,0].set_ylabel('numaxG')
    axs[1,0].set_ylabel('KSPG')
    axs[2,0].set_ylabel('KiPG')
    plt.show()


def main():
    # singleRun()
    localSensitivityAnalysis()

if __name__=='__main__':
    main()