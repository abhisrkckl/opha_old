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
M = 18498130000.*MSun
eta = 0.008
Pb0 = 12.18/(1+z)/1.107
nb0 = 2*np.pi/(9*year)
x0  = (M*nb0)**(2./3)
e0  = 0.67
u0  = 0.1
t0  = 1886.623*year

params_true = [x0, e0, u0, t0, M, eta]

data = np.genfromtxt('OJ287_sim.txt')
data_x = np.pi*np.round(data[:,0])
data_y = data[:,1]*year
data_yerr = data[:,2]*day

loglike = Likelihood(data_x, data_y, data_yerr, z) 
# loglike(params)
# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
x0*=1.042
xmin,xmax = x0*0.99, x0*1.05
emin,emax = 0.6435,0.656
umin,umax = 0.32,0.35
t0min,t0max = t0-30*day,t0+30*day
Mmin,Mmax = M*0.97, M*1.04
etamin,etamax = 0.004,0.016
mins = np.array((xmin,emin,umin,t0min,Mmin,etamin))
maxs = np.array((xmax,emax,umax,t0max,Mmax,etamax))
spans = maxs-mins
def prior_transform(x):
	return spans*x + mins
    
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


outburst_time_samples = BinX_PN_py.outburst_times_x(result.samples, data_x, 1e-14, 1e-14, 0.1)
outburst_time_samples = (t0 + (outburst_time_samples-t0)*(1+z))/year
def plot_outburst_time_dists(outburst_time_samples, n_per_row=5, bins=50, wt_cutoff=1e-30):
	n_outbursts = outburst_time_samples.shape[1]
	n_rows = (n_outbursts//n_per_row) + (1 if n_outbursts%n_per_row>0 else 0)
	
	idxs = result.weights>wt_cutoff
	_weights = result.weights[idxs]
	for n_cell in range(n_outbursts):
		_samples = (outburst_time_samples[:,n_cell])[idxs]
		plt.subplot(n_rows, n_per_row, n_cell+1)
		plt.hist(_samples, weights=_weights, bins=bins)
		plt.axvline(x=data_y[n_cell]/year, color='red')
	plt.show()
plot_outburst_time_dists(outburst_time_samples)
	
