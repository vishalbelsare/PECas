# -*- coding: utf-8 -*-

# ex6.py: Example 6 shows how to estimate the parameters of a general system
# described by an ODE with no analitical solution using the Euler's method.
# Furthermore, it illustrates a parametric identification where two different
# kind of measurements (angle and angular speed) are used, which have different
# standar deviation.

import pecas as pc
import pylab as pl
import casadi as ca

def mainEx6EIMX():
    data = pl.loadtxt('ex6data.txt')
    
    t = data[:, 0]
    phim = data[:, 1]
    wm = data[:, 2]
    N = t.size
    
    m = 1  # Ball mass in Kg
    L = 3  # Rod length in meters
    g = 9.81  # Gravity constant in m/s^2
    psi = pl.pi/2  # Handler (actuation) angle in radians
    
    dT = ca.MX.sym("dT", 1)
    phip = ca.MX.sym("phip", 1)
    wp = ca.MX.sym("wp", 1)
    
    # Theta = [Phi0,W0,K]
    Theta = ca.MX.sym("Theta", 3)
    
    # State x = [phi,w]
    
    # simstep
    
    k1 = (Theta[2]/(m*(L**2))*(psi-(phip)) - g/L * pl.sin(phip))
    k2 = (Theta[2]/(m*(L**2))*(psi-(phip + dT*k1/2)) - g/L * pl.sin(phip +
          dT*k1/2))
    k3 = (Theta[2]/(m*(L**2))*(psi-(phip + dT*k2/2)) - g/L * pl.sin(phip +
          dT*k2/2))
    k4 = (Theta[2]/(m*(L**2))*(psi-(phip + dT*k3)) - g/L * pl.sin(phip +
          dT*k3))

    ws = wp + (k1+2*k2+2*k3+k4)*dT/6
    phis = phip + wp*dT

    fphiw = ca.MXFunction([Theta, phip, wp, dT], [phis, ws])
    fphiw.setOption("name", "fphiw")
    fphiw.init()
    
    
    # simsloop
    
    Model = ca.MX.zeros(2*N, 1)    
    Model[0] = Theta[0]
    Model[t.size] = Theta[1]
    
    for k_phi in xrange(1, N):
        k_w = t.size+k_phi
        simfun = fphiw([Theta, Model[k_phi-1], Model[k_w-1],
                        t[k_phi] - t[k_phi-1]])
        Model[k_phi] = simfun[0]
        Model[k_w] = simfun[1]
    
    sigmaphi = pl.ones(t.size)*pl.std(phim, ddof=1)
    sigmaw = pl.ones(t.size)*pl.std(wm, ddof=1)
    sigma = pl.concatenate([sigmaphi, sigmaw])
    xinit = pl.ones(3)
    
    pep = pc.PECasLSq(Theta, Model, sigma, Y=pl.concatenate([phim, wm]),
                      xinit=xinit)
    pep.run_parameter_estimation()
    
    print pep.get_xhat()
    
    pep.compute_covariance_matrix()
    pep.print_results()
    return
    
if __name__ == "__main__":
    mainEx6EIMX()
