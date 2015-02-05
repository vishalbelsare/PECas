#!/usr/bin/python

import pecas as pc
import pylab as pl
import casadi as ca

data = pl.loadtxt('ex4p1data.txt')

t = data[:, 0]
vxm = data[:, 1]
Dk = data[:, 2]

D = ca.SX.sym("D", 1)
dT = ca.SX.sym("dT", 1)
vxp = ca.SX.sym("vxb", 1)

C1 = 10

C2 = ca.SX.sym("C2", 1)
C3 = ca.SX.sym("C3", 1)
vx0 = ca.SX.sym("vx0", 1)

# simstep

vx = (C1 * D - C2) * (1 / C3) * (1 - pl.exp(-C3 * dT)) + \
      vxp * pl.exp(-C3 * dT)

fvx = ca.SXFunction([C2, C3, vxp, dT, D], [vx])
fvx.setOption("name", "fvx")
fvx.init()

# simloop

# Theta = ca.SX.sym("Theta", 3)
Theta = ca.SX.sym("Theta", 3)

simvx = ca.SX.sym("simvx",t.size)

simvx[0] = Theta[2]

for k in xrange(1, t.size):

    simvx[k] = fvx([Theta[0], Theta[1], simvx[k-1], t[k] - t[k-1], Dk[k-1]])[0]


sigma = pl.ones(t.size)
xinit = pl.ones(3)

pep = pc.PECasProb(Theta, simvx, sigma, Y = vxm, xinit = xinit)
pep.run_parameter_estimation()

fsimvx = ca.SXFunction(ca.nlpIn(x = Theta), ca.nlpOut(f =simvx))
fsimvx.init()

fsimvx.setInput(pep.get_xhat(), "x")
fsimvx.evaluate()

fsim = fsimvx.getOutput("f")

pl.scatter(t, vxm)
pl.plot(t, fsim)
pl.show()

# b)

# dMdx = fsimvx.jac("x", "f")

# A = (vxm - simvx)
# Sigma_x = ca.mul(A.T, A) / (pep.get_N() - pep.get_d()) * \
#           ca.solve(ca.mul(dMdx.T, dMdx), pl.eye(pep.get_d()))

# Sigma_x = pep.get_Rhat() / (pep.get_N() - pep.get_d()) * \
#           ca.solve(ca.mul(dMdx.T, dMdx), pl.eye(pep.get_d()))

# fSigma_x = ca.SXFunction(ca.nlpIn(x = Theta), ca.nlpOut(f= Sigma_x))
# fSigma_x.init()

# fSigma_x.setInput(pep.get_xhat(), "x")
# fSigma_x.evaluate()

# print pl.array(fSigma_x.getOutput("f"))

pep.compute_covariance_matrix()
pep.print_results()

# from mpl_toolkits.mplot3d.axes3d import Axes3D

# w, v = pl.linalg.eig(pep.get_Covx())

# print w
# print v

# D = pl.diag(w)
# V = v

# xy=pl.array([pl.cos(pl.linspace(0,2*pl.pi,100)), pl.sin(pl.linspace(0,2*pl.pi,100)), pl.sin(pl.linspace(0,2*pl.pi,100))])

# ellipse = ca.mul(pep.get_xhat(), pl.ones([1,100])) +  ca.mul([V, D, xy])

# pl.subplot(3, 1, 1)
# pl.plot(pl.array(ellipse[0,:]).T, pl.array(ellipse[1,:]).T, color = 'b', label = "C2-C3")
# pl.scatter(pep.get_xhat()[0], pep.get_xhat()[1], color='b')
# pl.legend(loc="upper right")

# pl.subplot(3, 1, 2)
# pl.plot(pl.array(ellipse[0,:]).T, pl.array(ellipse[2,:]).T, color = 'r', label = "C2-vx0")
# pl.scatter(pep.get_xhat()[0], pep.get_xhat()[2], color='r')
# pl.legend(loc="upper left")

# pl.subplot(3, 1, 3)
# pl.plot(pl.array(ellipse[1,:]).T, pl.array(ellipse[2,:]).T, color = 'g', label = "C3-vx0")
# pl.scatter(pep.get_xhat()[1], pep.get_xhat()[2], color='g')
# pl.legend(loc="lower left")

# pl.show()

pep.plot_confidence_ellipsoids()