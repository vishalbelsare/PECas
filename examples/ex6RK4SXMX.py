# -*- coding: utf-8 -*-

# ex6.py: Example 6 shows how to estimate the parameters of a general system
# described by an ODE with no analitical solution using the Euler's method.
# Furthermore, it illustrates a parametric identification where two different
# kind of measurements (angle and angular speed) are used, which have different
# standar deviation.

import pecas as pc
import pylab as pl
import casadi as ca

def mainEx6EISXMX():
    data = pl.loadtxt('ex6data.txt')
    
    t = data[:, 0]
    phim = data[:, 1]
    wm = data[:, 2]
    N = t.size
    
    m = 1  # Ball mass in Kg
    L = 3  # Rod length in meters
    g = 9.81  # Gravity constant in m/s^2
    psi = pl.pi/2  # Handler (actuation) angle in radians
    
    dT = ca.SX.sym("dT", 1)
    phip = ca.SX.sym("phip", 1)
    wp = ca.SX.sym("wp", 1)
    
    
    Phi0 = ca.SX.sym("Phi0", 1)
    W0 = ca.SX.sym("W0", 1)
    K = ca.SX.sym("K", 1)
    
    k1 = (K/(m*(L**2))*(psi-(phip)) - g/L * pl.sin(phip))
    k2 = (K/(m*(L**2))*(psi-(phip+dT*k1/2)) - g/L * pl.sin(phip + dT*k1/2))
    k3 = (K/(m*(L**2))*(psi-(phip+dT*k2/2)) - g/L * pl.sin(phip + dT*k2/2))
    k4 = (K/(m*(L**2))*(psi-(phip+dT*k3)) - g/L * pl.sin(phip + dT*k3))    
    
    # simstep
    phis = phip + wp*dT
    ws = wp + (k1+2*k2+2*k3+k4)*dT/6
    
    fphiw = ca.SXFunction([K, phip, wp, dT], [phis, ws])
    fphiw.setOption("name", "fphiw")
    fphiw.init()
    
    # simsloop
    Theta = ca.MX.sym("Theta", 3)
    
    simphi = []
    simw = []
    simphi.append(Theta[0])    
    simw.append(Theta[1])    

    for k in xrange(1, t.size):
        simfun = fphiw.call([Theta[2], simphi[k-1], simw[k-1],
                        t[k] - t[k-1]])
        simphi.append(simfun[0])
        simw.append(simfun[1])
    
    simphi.extend(simw)
    simphiw = ca.vertcat(simphi)  
    sigmaphi = pl.ones(t.size)*pl.std(phim, ddof=1)
    sigmaw = pl.ones(t.size)*pl.std(wm, ddof=1)
    sigma = pl.concatenate([sigmaphi, sigmaw])
    xinit = pl.ones(3)
    
    pep = pc.PECasLSq(Theta, simphiw, sigma, Y=pl.concatenate([phim, wm]),
                      xinit=xinit)
    pep.run_parameter_estimation()
    
    print pep.get_xhat()
    
    pep.compute_covariance_matrix()
    pep.print_results()
    return
    
if __name__ == "__main__":
    mainEx6EISXMX()
