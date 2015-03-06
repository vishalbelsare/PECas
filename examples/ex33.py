import casadi as ca
import pylab as pl
import pecas

x = ca.SX.sym("x", 2)

alpha = ca.SX.sym("alpha", 1)
beta = ca.SX.sym("beta", 1)
gamma = ca.SX.sym("gamma", 1)
delta = ca.SX.sym("delta", 1)

p = ca.SX.sym("p", 4)

for k, pk in enumerate([alpha, beta, gamma, delta]):
    p[k] = pk

u = ca.SX.sym("u", 0)

f = ca.vertcat( \
    [-alpha * x[0] + beta * x[0] * x[1], 
    gamma * x[1] - delta * x[0] * x[1]])

y = x

op = pecas.systems.ExplODE(y = y, x = x, u = u, p = p, f = f)

data = pl.array(pl.loadtxt("data_ex33.txt"))

timegrid = data[:, 0]
yN = data[:, 1::2]
stdyN = data[:, 2::2]

pmin = pl.array([1.0, -pl.inf, 1.0, -pl.inf])
print pmin

xmin = -pl.inf*pl.ones((11, 2)).T

U = pl.array([])

odesol = pecas.setupmethods.ODEsetup( \
    system = op, timegrid = timegrid, \
    x0min = [yN[0,0], yN[0,1]], \
    x0max = [yN[0,0], yN[0,1]], \
    xmin = xmin, \
    umin = U, umax = U, uinit = U, \
    pmin = pmin, \
    pmax = [1.0, pl.inf, 1.0, pl.inf], \
    pinit = [1.0, 0.5, 1.0, 1.0])


lsqprob = pecas.LSq(pesetup=odesol, yN=yN, stdyN = stdyN)
lsqprob.run_parameter_estimation()

phat = odesol.V()(lsqprob.Vhat)["P"]
print phat

pl.scatter(timegrid, yN[:,0], color = 'b')
pl.scatter(timegrid, yN[:,1], color = 'r')

pl.plot(timegrid, ca.vertcat(odesol.V()(lsqprob.Vhat)["X",:,0])[::2], \
    color='b', ls = '--')
pl.plot(timegrid, ca.vertcat(odesol.V()(lsqprob.Vhat)["X",:,0])[1::2], \
    color='r', ls = '--')

tgridint = pl.linspace(0,10,1000)
ode = ca.SXFunction(ca.daeIn(x=x, p=p), ca.daeOut(ode=f))
integrator = ca.Integrator("cvodes", ode)

simulator = ca.Simulator(integrator, tgridint)
simulator.init()

simulator.setInput([1, 1], "x0")
simulator.setInput(phat, "p")
simulator.evaluate()

pl.plot(tgridint, pl.squeeze(simulator.getOutput("xf")[0,:]), color='b')
pl.plot(tgridint, pl.squeeze(simulator.getOutput("xf")[1,:]), color='r')
pl.grid()
pl.show()
