from __future__ import print_function, division, unicode_literals
import numpy as np
import nestle
import matplotlib.pyplot as plt
import importlib
import corner
#import sklearn.neighbors as skln
import awkde

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

epsabs, epsrel, init_step = 1e-14, 1e-14, 0.1

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



def kde_likelihood_fn(setup, z, phiobs, bws, sampless):
    kdes = []
    for bw, samples in zip(bws, sampless):
        #kde = skln.KernelDensity(bandwidth=bw)
        kde = awkde.GaussianKDE(glob_bw='silverman', alpha=0.4)
        kde.fit(samples[:,np.newaxis])
        kdes.append(kde)
    
    #modelsetup = model.ModelSetup(phiobs, z, epsabs, epsrel, init_step)
    
    def kde_lnlikelihood(params):
        tobs_model = setup.outburst_times_E(params)
        #return sum([ kde.score_samples([[tob_model]])[0] for kde, tob_model in zip(kdes, tobs_model) ])
        return sum([ kde.score(tob_model) for kde, tob_model in zip(kdes, tobs_model) ])
   
    return kde_lnlikelihood

def read_kde_files(kde_file, tobs_samples_file_fmt):
    nobs, yrs, bws = np.genfromtxt(kde_file).transpose()
    
    sampless = [year*np.genfromtxt(tobs_samples_file_fmt.format(int(yr))) for yr in yrs]
    
    return np.pi*nobs, bws, sampless

class ModelSetup:
    
    def __init__(self, model_name, prior_file, outbursts_file, tobs_samples_file_fmt, like_type="Gaussian", z=0.306):
        self.model = get_model(model_name)
        self.ndim = self.model.N_PARAMS
        self.like_type = like_type
        self.prior_transform = read_prior_transform(prior_file, self.ndim)
        self.redshift = z
        self.param_names = self.model.param_names().split(',')
        
        self.phiobs, self.bws, self.sampless = read_kde_files(outbursts_file, tobs_samples_file_fmt)
        
        if like_type == "Gaussian":
            
            #self.phiobs, self.tobs, self.tob_errs = get_tobs_data(tobs_file)
            
            self.tobs = np.array( [np.mean(samples) for samples in self.sampless] )
            self.tob_errs = np.array( [np.std(samples) for samples in self.sampless] )
            
            self.lnlike = self.model.Likelihood(self.phiobs, self.tobs, self.tob_errs, z)
        
        elif like_type=="KDE":
            
            self.setup = self.model.ModelSetup(self.phiobs, z, epsabs, epsrel, init_step)
            self.outburst_times = self.setup.outburst_times_E
            
            self.sample_mins = [min(samples) for samples in self.sampless]
            self.sample_maxs = [max(samples) for samples in self.sampless]
            self.Nsamples = [len(samples) for samples in self.sampless]
            
            self.kdes = []            
            for bw, samples in zip(self.bws, self.sampless):
                kde = awkde.GaussianKDE(glob_bw=bw, alpha=0.4)
                kde.fit(samples[:,np.newaxis])
                self.kdes.append(kde)
            
            def score(kde, tob_model, tob_min, tob_max, bw, N):
                if tob_model < tob_min:
                    return -0.5*((tob_model-tob_min)/bw)**2 - np.log(np.sqrt(2*np.pi)*bw*N)
                elif tob_model > tob_max:
                    return -0.5*((tob_model-tob_max)/bw)**2 - np.log(np.sqrt(2*np.pi)*bw*N)
                else:
                    return kde.score(tob_model)
            
            def scores(params):
                tobs_model = self.outburst_times(params)
                return [ score(kde, tob_model, tob_min, tob_max, kde.glob_bw, N) for kde, tob_model, tob_min, tob_max, N in zip(self.kdes, tobs_model, self.sample_mins, self.sample_maxs, self.Nsamples) ]
                
            self.scores = scores
            
            def lnlike(params):
                return sum(scores(params))
            
            self.lnlike = lnlike # kde_likelihood_fn(self.setup, z, self.phiobs, self.bws, self.sampless)
            
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

def print_results(modelsetup, result, units, shifts, unit_strs):
    means, covs = nestle.mean_and_cov(result.samples, weights=result.weights)
    stds = np.diag(covs)**0.5
    
    for pname, mean, std, uname in zip(modelsetup.param_names, means, stds, unit_strs):
        print("{} = {} +/- {}  {}".format(pname, mean, std,uname))
    

if __name__=='__main__':

    #modelsetup1 = ModelSetup("Model6", "nospin_priors.txt", tobs_file="OJ287_1templ.txt")
    #modelsetup1 = ModelSetup("Model8", "spin_priors.txt", tobs_file="OJ287_1templ.txt")
    modelsetup1 = ModelSetup("Model8", "spin_priors.txt", "OJ287_1templ_bandwidths.txt", tobs_samples_file_fmt="../../OJ287-lightcurve/oj287_tobs_samples_1templ_{}.txt", like_type="Gaussian")
    
    result = modelsetup1.run_sampler()

    units, unames, shifts = read_plot_settings('spin_plot_settings.txt')
    plot_posterior(modelsetup1, result, units=units, shifts=shifts, unit_strs=unames)
    plot_residuals(modelsetup1, result)

    print()
    
    print_results(modelsetup1, result, units, shifts, unames)
    
    


