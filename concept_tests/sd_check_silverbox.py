import casadi as ca
import pylab as pl
import pecas

N = 1000
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
tu = pl.linspace(0, N, N+1) * dt

uN = ca.DMatrix(0.1*pl.random(N))

yN = pl.zeros((x.shape[0], N+1))
wv = pl.ones(yN.shape)

lsqpe_sim = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = p_guess, \
    xinit = yN, 
    # linear_solver = "ma97", \
    yN = yN, wv = wv)

lsqpe_sim.run_simulation(x0 = [0.0, 0.0], psim = p_true/scale)

p_test = []

sigma = 0.01
wv = (1. / sigma**2) * pl.ones(yN.shape)

for k in range(200):

    y_test = lsqpe_sim.Xsim + sigma * (pl.randn(*lsqpe_sim.Xsim.shape))

    lsqpe_test = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = p_guess, \
    xinit = y_test, 
    linear_solver = "ma97", \
    yN = y_test, wv = wv)

    lsqpe_test.run_parameter_estimation()

    p_test.append(lsqpe_test.phat)


p_mean = []
p_std = []

for j, e in enumerate(p_true):

    p_mean.append(pl.mean([k[j] for k in p_test]))
    p_std.append(pl.std([k[j] for k in p_test], ddof = 1))


print pl.asarray(p_mean)
print pl.asarray(p_std)

lsqpe_test.compute_covariance_matrix()
print lsqpe_test.phat
print ca.diag(ca.sqrt(lsqpe_test.Covp))

