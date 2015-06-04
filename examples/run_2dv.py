import casadi as ca
import pylab as pl
import pecas

# System

x = ca.MX.sym("x", 4)
p = ca.MX.sym("p", 6)
u = ca.MX.sym("u", 2)
we = ca.MX.sym("we", 4)

f = ca.vertcat( \

    [x[3] * pl.cos(x[2] + p[0] * u[0]) + we[0],

    x[3] * pl.sin(x[2] + p[0] * u[0]) + we[1],

    x[3] * u[0] * p[1] + we[2],

    p[2] * u[1] \
        - p[3] * u[1] * x[3] \
        - p[4] * x[3]**2 \
        - p[5] \
        - (x[3] * u[0])**2 * p[1] * p[0] + we[3]])

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, we = we, f = f, y = y)

# Inputs

data = pl.array(pl.loadtxt( \
    "controlReadings_ACADO_MPC_Betterweights.dat", \
    delimiter = ", ", skiprows = 1))

tu = data[300:400, 1]

yN = data[300:400, [2, 4, 6, 8]]
wv = 1 / (0.1**2) * pl.ones(yN.shape)
uN = data[300:399, [9, 10]]
wwe = [1 / 1e-4] * 4

porig = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

# Run parameter estimation and assure that the results is correct

lsqpe = pecas.LSq(system = odesys, \
    tu = tu, u = uN, \
    pinit = [0.5, 17.06, 11.5, 5, 0.07, 0.70], \
    yN =yN, wv = wv, wwe = wwe)

lsqpe.show_system_information(showEquations = True)

lsqpe.run_parameter_estimation()
lsqpe.show_results()

lsqpe.compute_covariance_matrix()
lsqpe.show_results()

xhat = lsqpe.Xhat[0]
yhat = lsqpe.Xhat[1]
psihat = lsqpe.Xhat[2]
vhat = lsqpe.Xhat[3]

pl.close("all")

pl.figure()
pl.subplot(4, 1, 1)
pl.plot(xhat)
pl.plot(yN[:,0])

pl.subplot(4, 1, 2)
pl.plot(yhat)
pl.plot(yN[:,1])

pl.subplot(4, 1, 3)
pl.plot(psihat)
pl.plot(yN[:, 2])

pl.subplot(4, 1, 4)
pl.plot(vhat)
pl.plot(yN[:, 3])

pl.figure()
pl.plot(xhat, yhat)
pl.plot(yN[:,0], yN[:, 1])

pl.show()
