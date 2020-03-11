from run_model import *
import numpy as np

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

model_name = "Model9"
model = get_model(model_name)

data_file = "oj287_new1.txt"
data = get_data(data_file)

# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
M = 20000000000.*MSun
t0  = 1886.05*year
x0min,x0max = 0.0180, 0.0210
e0min,e0max = 0.635,0.645
u0min,u0max = 0.37,0.40
t0min,t0max = 1885.9*year,1886.3*year
Mmin,Mmax = 21.5e9*MSun, 27.6e9*MSun
etamin,etamax = 0.012,0.0165
Ximin,Ximax = 0.40,0.90
d1min,d1max = 0.30,0.50
#ddmin,ddmax = 0.018,0.036
mins = np.array((x0min,e0min,u0min,t0min,Mmin,etamin,Ximin,d1min))
maxs = np.array((x0max,e0max,u0max,t0max,Mmax,etamax,Ximax,d1max))
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
           ("$\\Xi$",       1,  "",    0),
           ("$d_{em}$",     1,     "",            0),
          #("$d_{dd}$",     1,     "",            0),
          #("$d_{dc}$",     1,     "",            0)
         ]
print_results(result, display_params)
plot_posterior(model, data, result, display_params, model_name+"_post", nbins=15, z=0.306)


