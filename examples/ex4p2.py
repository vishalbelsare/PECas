#!/usr/bin/python

import pecas as pc
import pylab as pl
import casadi as ca

data = pl.loadtxt('ex4p2data.txt')

t = data[:, 0]
vxm = data[:, 1]
Dk = data[:, 2]

D = ca.SX.sym("D", 1)
dT = ca.SX.sym("dT", 1)
vxp = ca.SX.sym("vxb", 1)

C1 = ca.SX.sym("C1", 1)

C2 = ca.SX.sym("C2", 1)
C3 = ca.SX.sym("C3", 1)
vx0 = ca.SX.sym("vx0", 1)

# simstep

vx = (C1 * D - C2) * (1 / C3) * (1 - pl.exp(-C3 * dT)) + \
    vxp * pl.exp(-C3 * dT)

fvx = ca.SXFunction([C1, C2, C3, vxp, dT, D], [vx])
fvx.setOption("name", "fvx")
fvx.init()

# simloop

Theta = ca.SX.sym("Theta", 4)

simvx = ca.SX.sym("simvx", t.size)
simvx[0] = fvx([Theta[0], Theta[1], Theta[2], Theta[3], 0, Dk[0]])[0]

for k in xrange(1, t.size):

    simvx[k] = fvx([Theta[0], Theta[1], Theta[2], simvx[k-1],
                   t[k] - t[k-1], Dk[k]])[0]


sigma = pl.ones(t.size)
xinit = pl.ones(4)

pep = pc.PECasProb(Theta, simvx, sigma, Y=vxm, xinit=xinit)
pep.run_parameter_estimation()

print pep.get_xhat()

fsimvx = ca.SXFunction([Theta], [simvx])
fsimvx.init()

fsimvx.setInput(pep.get_xhat())
fsimvx.evaluate()

fsim = fsimvx.getOutput()

pl.scatter(t, vxm)
pl.plot(t, fsim)
pl.show()

pep.compute_covariance_matrix()
pep.print_results()

pep.plot_confidence_ellipsoids()
