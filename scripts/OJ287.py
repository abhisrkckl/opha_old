from BinX_PN_py import outburst_times
import numpy as np
import matplotlib.pyplot as plt

day = 24*3600
year = 365*day
MSun = 4.92703806e-6

#true params
M = 1.84e10*MSun
eta = 0.01
nb0 = 2*np.pi/(8.34*year)
x0 = (M*nb0)**(2./3)
e0 = 0.65
u0 = np.pi/6
t0  = 1886.605*year

params_true = [x0, e0, u0, t0, M, eta]

data = np.genfromtxt('OJ287_sim.txt')
data_x = np.pi*np.round(data[:,0])
data_y = data[:,1]*year
data_yerr = data[:,2]*day


