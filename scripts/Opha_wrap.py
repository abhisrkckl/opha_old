from __future__ import print_function, division
import numpy as np
import nestle
import matplotlib.pyplot as plt
import importlib
import corner

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

def get_model(model_name):
    model_pkg = model_name+'_py'
    model = importlib.import_module(model_pkg)
    return model

def get_tobs_data(data_file="OJ287_real.txt"):
    data = np.genfromtxt(data_file, comments='#')
    data_x = np.pi*np.round(data[:,0])
    data_y = data[:,1]*year
    data_yerr = data[:,2]*year
    return data_x, data_y, data_yerr

def read_prior_transform(prior_file, ndim):
    mins, maxs, units = np.genfromtxt(prior_file, usecols=(1,2,3)).transpose()
    mins *= units
    maxs *= units
    spans = maxs-mins
    
    if len(spans) != ndim:
        raise ValueError("Incorrect number of columns in {}. ncols={}, ndim={}".format(prior_file, len(spans), ndim))
    
    def prior_transform(cube):
        return spans*cube + mins
    
    return prior_transform

class ModelSetup:
    
    def __init__(self, model_name, prior_file, like_type="Gaussian", tobs_file=None, kde_file=None, tobs_samples_file_fmt=None, z=0.306):
        self.model = get_model(model_name)
        self.ndim = self.model.N_PARAMS
        self.like_type = like_type
        self.prior_transform = read_prior_transform(prior_file, self.ndim)
        self.redshift = z
        self.param_names = self.model.param_names().split(',')
        
        if like_type == "Gaussian" and tobs_file is not None:
            self.phiobs, self.tobs, self.tob_errs = get_tobs_data(tobs_file)
            self.lnlike = self.model.Likelihood(self.phiobs, self.tobs, self.tob_errs, z)
        
        elif like_type=="KDE" and kde_file is not None and tobs_samples_file_fmt is not None:
            pass
            
        else:
            raise ValueError("The options are invalid.")
    
    def run_sampler(self, npoints=100, method='multi', print_progress=True):
        print("Model Description : ", self.model.description())
        print("Model Parameters : ", self.model.param_names())
        callback = nestle.print_progress if print_progress else None
        return nestle.sample(self.lnlike, self.prior_transform, self.ndim, npoints=npoints, method=method, callback=callback)

def corner_label(pname, uname, shift):
    label = "\n${}$".format(pname)
    if uname not in [None, "", '_']: 
        label = "{} ({})".format(label, uname)
    
    if shift>0:
        label = "{} - {}".format(label, shift)
    elif shift<0:
        label = "{} + {}".format(label, -shift)    
    
    return label

def plot_posterior(modelsetup, sampler_result, units, shifts, unit_strs):
    
    corner_labels = [ corner_label(pname, uname, shift) for pname, uname, shift in zip(modelsetup.param_names, unit_strs, shifts) ]
    
    scaled_samples = sampler_result.samples/units - shifts
    
    corner.corner(scaled_samples, weights=result.weights, labels=corner_labels, label_kwargs = {"labelpad":100, "fontsize":12}, range=[1-1e-15]*modelsetup.ndim)
    plt.show()

def plot_residuals(modelsetup, sampler_result):
    means, covs = nestle.mean_and_cov(sampler_result.samples, sampler_result.weights)
    tobs_model = modelsetup.model.outburst_times_E(means, modelsetup.phiobs, modelsetup.redshift, 1e-14, 1e-14, 0.1)
    plt.errorbar(modelsetup.tobs/year, (modelsetup.tobs-tobs_model)/day, yerr=modelsetup.tob_errs/day, fmt='ro')
    plt.grid()
    plt.xlabel("Outburst Time (yr)")
    plt.ylabel("Residual (days)")
    plt.show()

def read_plot_settings(filename):
    settings = np.genfromtxt(filename, dtype=("S10", float,'S20',float)).transpose()
    units = np.array([ row[1] for row in settings ])
    unames = [ row[2] for row in settings ]
    shifts = np.array([ row[3] for row in settings ])
    
    return units, unames, shifts

modelsetup = ModelSetup("Model6", "nospin_priors.txt", tobs_file="OJ287_1templ.txt")
result = modelsetup.run_sampler()

units, unames, shifts = read_plot_settings('nospin_plot_settings.txt')
plot_posterior(modelsetup, result, units=units, shifts=shifts, unit_strs=unames)
plot_residuals(modelsetup, result)


print()


