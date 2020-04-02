from run_model import *
import numpy as np

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

model_name = "NoSpin"
model = get_model(model_name)

data_file = "oj287_new1.txt"
data = get_data(data_file)

# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
M = 20000000000.*MSun
t0  = 1886.05*year
x0min,x0max = 0.017, 0.018
e0min,e0max = 0.620,0.647
u0min,u0max = 0.30,0.45
t0min,t0max = t0-35*day,t0+350*day
Mmin,Mmax = M*0.82, M*1.08
etamin,etamax = 0.0001,0.032
d1min,d1max = 0.4,0.85
ddmin,ddmax = 0,2.3e6/(365.25*24*3600)
mins = np.array((x0min,e0min,u0min,t0min,Mmin,etamin,d1min,ddmin))
maxs = np.array((x0max,e0max,u0max,t0max,Mmax,etamax,d1max,ddmax))
spans = maxs-mins
def prior_transform(x):
        return spans*x + mins

z=0.306

result = run_sampler(model, prior_transform, data, z, npts=100)

display_params = [ ("$x_0$",         1e-2,     "$10^{-2}$",        0),
           ("$e_0$",         1,     "",            0),
           ("$u_0$",         1,     "rad",            0),
           ("$t_0$",         year,     "yr",            1886),
           ("$M$",         1e9*MSun,"$10^9 M_{Sun}$",    0),
           ("$\\eta$",         1e-2,     "$10^{-2}$",        0),
           ("$d_{em}$",     1,     "",            0),
           ("$d_{dd}$",     1,     "",            0),
           #("$d_{dc}$",     1,     "",            0)
         ]
print_results(result, display_params)
plot_posterior(model, data, result, display_params, model_name+"_post", nbins=15, z=0.306)


#plot_residuals(model_name, result, data_file='OJ287_real.txt')
"""
samples_new = nestle.resample_equal(result.samples,result.weights)
outburst_time_samples = model.outburst_times_x(samples_new, data[0], 1e-14, 1e-14, 0.1)
t0s = samples_new[:,3]
outburst_time_samples_yr = np.zeros_like(outburst_time_samples)
for idx,(tob_sample, t0) in enumerate(zip(outburst_time_samples,t0s)):
    outburst_time_samples_yr[idx] = (t0 + (tob_sample-t0)*(1+z))/year
#outburst_time_samples = (t0s + (outburst_time_samples-t0s)*(1+z))/year
data_y = data[1]

def plot_outburst_time_dists(outburst_time_samples, n_per_row=5, bins=50, wt_cutoff=1e-30):
    n_outbursts = outburst_time_samples.shape[1]
    n_rows = (n_outbursts//n_per_row) + (1 if n_outbursts%n_per_row>0 else 0)
    
    idxs = result.weights>wt_cutoff
    _weights = result.weights[idxs]
    for n_cell in range(n_outbursts):
        _samples = (outburst_time_samples[:,n_cell])[idxs]
        plt.subplot(n_rows, n_per_row, n_cell+1)
        plt.hist(_samples, bins=8)
        plt.axvline(x=data_y[n_cell]/year, color='red')
    plt.show()
plot_outburst_time_dists(outburst_time_samples)    """
