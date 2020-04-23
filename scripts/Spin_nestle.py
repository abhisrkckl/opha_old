import matplotlib
matplotlib.use('Agg')

from run_model import *
from ModelSetup import prior_transform_fn, lnlike_fn
import numpy as np

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

model_name = "Spin"
model = get_model(model_name)

data_dir = "../data/outburst-times/"
data_file = "{}/oj287_data_new1.txt".format(data_dir)
data = get_data(data_file)

z=0.306

lnlike = lnlike_fn(model, z, datafile=data_file)
prior_transform = prior_transform_fn("../data/config/spin_priors.txt")

result = nestle.sample(lnlike, prior_transform, model.N_PARAMS, npoints=100, method='multi', callback=nestle.print_progress)

display_params = [  ("$x_0$",         1e-2,     "$10^{-2}$",        0),
                    ("$e_0$",         1,        "",                 0),
                    ("$u_0$",         1,        "rad",              0),
                    ("$t_0$",         year,     "yr",               1886),
                    ("$M$",           1e9*MSun, "$10^9 M_{Sun}$",   0),
                    ("$\\eta$",       1e-2,     "$10^{-2}$",        0),
                    ("$\\Xi$",        1,        "",                 0),
                    ("$d_{em}$",      1,        "",                 0),
                    ("$d_{dd}$",      1,        "",                 0),     ]
                    
print_results(result, display_params)
plot_posterior(model, data, result, display_params, model_name+"_post", nbins=15, z=0.306)

