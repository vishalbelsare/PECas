import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    # return a * pl.exp(-b * x) + c
    return a ** (x - b) + c

x = pl.array([10, 50, 80, 100, 150, 180])
yn = pl.array([1, 21, 132, 218, 652, 1516])

popt, pcov = curve_fit(func, x, yn)

pl.cla()
pl.plot(x, yn)
pl.plot(x, func(x, *popt))
pl.show()