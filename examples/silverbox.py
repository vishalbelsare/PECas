import pylab as pl

import casadi as ca
import pecas

N = 10000
fs = 610.1

p_true = ca.DMatrix([5.625e-6,2.3e-4,1,4.69])
p_guess = ca.DMatrix([5,3,1,5])
scale = ca.vertcat([1e-6,1e-4,1,1])

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)
p = ca.MX.sym("p", 4)

f = ca.vertcat([
        x[1], \
        (u - scale[3] * p[3] * x[0]**3 - scale[2] * p[2] * x[0] - \
            scale[1] * p[1] * x[1]) / (scale[0] * p[0]), \
    ])

phi = x

odesys = pecas.systems.ExplODE( \
    x = x, u = u, p = p, f = f, phi = phi)

dt = 1.0 / fs
t = pl.linspace(0, N, N+1) * dt

u_data = ca.DMatrix(0.1*pl.random(N))

y_data_dummy = pl.zeros((x.shape[0], N+1))
wv_dummy = y_data_dummy

lsqpe_dummy = pecas.LSq( \
    system = odesys, \
    tu = t, \
    uN = u_data, \
    yN = y_data_dummy, \
    wv = wv_dummy)

lsqpe_dummy.run_simulation(x0 = [0.0, 0.0], psim = p_true/scale)

y_data = lsqpe_dummy.Xsim
y_data += 1e-3 * pl.random((x.shape[0], N+1))

wv = pl.ones(y_data.shape)

lsqpe = pecas.LSq( \
    system = odesys, \
    tu = t, \
    uN = u_data, \
    yN = y_data, \
    pinit = p_guess, \
    xinit = y_data, \
    linear_solver = "ma97", \
    wv = wv)

lsqpe.run_parameter_estimation()
lsqpe.run_simulation(x0 = [0.0, 0.0])

pl.close("all")
pl.figure()

pl.scatter(t, pl.squeeze(y_data[0,:]))
pl.plot(t, lsqpe.Xsim[0,:].T)

pl.scatter(t, pl.squeeze(y_data[1,:]))
pl.plot(t, lsqpe.Xsim[1,:].T)

pl.show()
