import numpy as np
import nestle
import corner
import matplotlib.pyplot as plt
from BinX_PN_py import Likelihood
from time import time

day = 24*3600
year = 365*day
MSun = 4.92703806e-6

#true params
z   = 0.306
M = 1.9e10*MSun
eta = 0.01
#nb0 = 2*np.pi/(8.34*year)
x0 =  1.7e-2 #(M*nb0)**(2./3)
e0 = 0.65
u0 = 0.34
t0  = 1886.623*year
params_true = [x0, e0, u0, t0, M, eta]

data = np.genfromtxt('OJ287_real.txt')
data_x = np.pi*np.round(data[:,0])
data_y = data[:,1]*year
data_yerr = data[:,2]*day

xmin,xmax = x0*0.95, x0*1.05
emin,emax = 0.63,0.66
umin,umax = 0.32,0.36
t0min,t0max = t0-40*day,t0+20*day
Mmin,Mmax = M*0.95, M*1.05
etamin,etamax = 0.004,0.016
mins = np.array((xmin,emin,umin,t0min,Mmin,etamin))
maxs = np.array((xmax,emax,umax,t0max,Mmax,etamax))
spans = maxs-mins
def prior_transform(x):
	return spans*x + mins

loglike = Likelihood(data_x, data_y, data_yerr, z)


# Run nested sampling.
ndim = len(params_true)
begin_time=time()
result = nestle.sample(loglike, prior_transform, ndim, npoints=1500)
print "Time elapsed = ",time()-begin_time

print "log z = ",result.logz     # log evidence
print "log z err = ",result.logzerr  # numerical (sampling) error on logz

samples = result.samples.copy()
xsamples = samples[:,0]
Msamples = samples[:,4]
nsamples = (xsamples**1.5)/Msamples
samples[:,3]/=year
samples[:,4]/=(1e9*MSun)
#samples[:,0]=1./(nsamples*year/(2*np.pi))
corner.corner(	samples, weights=result.weights,
		quantiles=[0.0455, 0.5, 0.9545], bins=20,
		labels=['$x$','$e_0$','$u_0$ (rad)','$t_0$ (y)','$M$ ($10^{9} M_{Sun}$)','$\eta$'],
		show_titles=True,
		title_fmt=".1e")
plt.show()
