import casadi as ca
import pylab as pl
import pecas

# System

x = ca.MX.sym("x", 4)
p = ca.MX.sym("p", 4)
u = ca.MX.sym("u", 2)
we = ca.MX.sym("we", 4)

f = ca.vertcat( \

    [x[3] * pl.cos(x[2] + 0.5 * u[0]) + we[0],

    x[3] * pl.sin(x[2] + 0.5 * u[0]) + we[1],

    x[3] * u[0] * 17.06 + we[2],

    12.0 * p[0] * u[1] \
        - 2.17 * p[1] * u[1] * x[3] \
        - 0.1 * p[2] * x[3]**2 \
        - 0.6 * p[3] \
        - (x[3] * u[0])**2 * 17.06 * 0.5 + we[3]])

phi = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, we = we, f = f, phi = phi)
odesys.show_system_information(showEquations = True)

# Inputs

data = pl.array(pl.loadtxt("data_2d_vehicle.dat", \
    delimiter = ", ", skiprows = 1))

ty = data[300:850, 1]

yN = data[300:850, [2, 4, 6, 8]]
# wv = 1 / (0.1**2) * pl.ones(yN.shape)
wv = pl.ones(yN.shape)
# wv[:, 0] = 10.0
# wv[:, 1] = 0.1
# wv[:,2] = 0.1
wv[:,3] = 0.1
uN = data[300:849, [9, 10]]
# wwe = [1 / 1e-1] * 4
# wwe = [1.0] * 2 + [0.1] + [0.5]
wwe = [10.0] * 2 + [1.0] * 2
# wwe[0] = 1 / 1e-3
# wwe[-1] = 1 / 1e-3
# wwe = [1 / 1e-2] * 3 + [1/1e-1]

porig = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

lsqpe = pecas.LSq(system = odesys, \
    tu = ty, uN = uN, \
    # pinit = [11.5, 2.0, 0.07, 0.70], \
    pinit = [1.0] * 4, \
    ty = ty, yN =yN, \
    wv = wv, wwe = wwe, \
    xinit = yN, \
    # linear_solver = "mumps", \
    linear_solver = "ma97", \
    scheme = "radau", \
    order = 3)

lsqpe.run_parameter_estimation(hessian = "exact-hessian")

# lsqpe.covmat_schur()
# var1 = lsqpe.Covp 

# lsqpe.covmat_backsolve()
# var2 = lsqpe.Covp[:4, :4]

# lsqpe.compute_covariance_matrix()
# var3 = lsqpe.Covp 

# reldev13 = var1 / var3
# reldev23 = var2 / var3
# reldev12 = var1 / var2

# lsqpe.show_results()

# lsqpe.compute_covariance_matrix()
# lsqpe.show_results()

# xhat = lsqpe.Xhat[0]
# yhat = lsqpe.Xhat[1]
# psihat = lsqpe.Xhat[2]
# vhat = lsqpe.Xhat[3]

lsqpe.run_simulation(x0 = yN[0,:])

xhat = lsqpe.Xsim[0,:].T
yhat = lsqpe.Xsim[1,:].T
psihat = lsqpe.Xsim[2,:].T
vhat = lsqpe.Xsim[3,:].T

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
