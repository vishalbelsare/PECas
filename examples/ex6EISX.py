# -*- coding: utf-8 -*-

# ex6.py: Example 6 shows how to estimate the parameters of a general system
# described by an ODE with no analitical solution using the Euler's method.
# Furthermore, it illustrates a parametric identification where two different
# kind of measurements (angle and angular speed) are used, which have different
# standar deviation.

import pecas as pc
import pylab as pl
import casadi as ca

def mainEx6EISX():
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
    
    # simstep
    phis = phip + wp*dT
    ws = wp + (K/(m*(L**2))*(psi-phip) - g/L * pl.sin(phip))*dT
    
    fphiw = ca.SXFunction([K, phip, wp, dT], [phis, ws])
    fphiw.setOption("name", "fphiw")
    fphiw.init()
    
    # simsloop
    Theta = ca.SX.sym("Theta", 3)
    
    simphiw = ca.SX.sym("simphiw", t.size*2)
    simphiw[0] = Theta[0]
    simphiw[t.size] = Theta[1]
    
    for k_phi in xrange(1, t.size):
        k_w = t.size+k_phi
        simfun = fphiw([Theta[2], simphiw[k_phi-1], simphiw[k_w-1],
                        t[k_phi] - t[k_phi-1]])
        simphiw[k_phi] = simfun[0]
        simphiw[k_w] = simfun[1]
    
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
    mainEx6EISX()
