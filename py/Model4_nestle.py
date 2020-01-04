from run_model import *
import numpy as np

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

model_name = "Model4"
model = get_model(model_name)

data_file = "oj287_new.txt"
data = get_data(data_file)

# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
M = 20000000000.*MSun
t0  = 1886.05*year
x0min,x0max = 0.0167, 0.0180
e0min,e0max = 0.62,0.66
u0min,u0max = 0.25,0.45
t0min,t0max = t0-210*day,t0+270*day
Mmin,Mmax = M*0.9, M*1.05
etamin,etamax = 0.014,0.06
d1min, d1max = 0.35,0.8
mins = np.array((x0min,e0min,u0min,t0min,Mmin,etamin,d1min))
maxs = np.array((x0max,e0max,u0max,t0max,Mmax,etamax,d1max))
spans = maxs-mins
def prior_transform(x):
        return spans*x + mins

result = run_sampler(model, prior_transform, data, z=0.306, npts=100)

display_params = [ ("$x_0$", 		1e-2, 	"$10^{-2}$",		0),
		   ("$e_0$", 		1, 	"",			0),
		   ("$u_0$", 		1, 	"rad",			0),
		   ("$t_0$",	 	year, 	"yr",			1886),
		   ("$M$", 		1e9*MSun,"$10^9 M_{Sun}$",	0),
		   ("$\\eta$", 		1e-2, 	"$10^{-2}$",		0),
		   ("$d_{em}$", 	1, 	"",			0)
		 ]
print_results(result, display_params)
plot_posterior(model, data, result, display_params, model_name+"_post", nbins=15, z=0.306)


#plot_residuals(model_name, result, data_file='OJ287_real.txt')

"""
# Run nested sampling.
ndim = len(params_true)
begin_time=time()
result = nestle.sample(loglike, prior_transform, ndim, npoints=300)
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
		#quantiles=[0.0455, 0.5, 0.9545], 
		bins=20,
		labels=['$x$','$e_0$','$u_0$ (rad)','$t_0$ (y)','$M$ ($10^{9} M_{Sun}$)','$\eta$','d1'],
		label_kwargs = {"labelpad":60, "fontsize":14},
		show_titles=True,
		range=[0.9999]*ndim,
		title_fmt="0.3f")
plt.savefig("posterior.pdf")
plt.show()

""
outburst_time_samples = outburst_times_x(result.samples, data_x, 1e-14, 1e-14, 0.1)
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
plot_outburst_time_dists(outburst_time_samples)	"""
