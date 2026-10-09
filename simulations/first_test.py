import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m = 1.0
k = 4.0
c = 0.5
x0 = 1.0
v0 = 0.0
def oscillator(t, y):
    x, v = y
