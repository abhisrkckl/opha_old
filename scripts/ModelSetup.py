import numpy as np
from sklearn.neighbors import KernelDensity

import Model6_py as NonSpinning
import Model8_py as Spinning

def uniform_prior_transform_fn(prior_lims):
    mins, maxs = np.asarray(prior_lims).transpose()
    spans = maxs-mins
    
    def prior_transform(cube):
        return spans*cube + mins
    
    return prior_transform

class OutburstTimes:
    def __init__(self, nobs, tobs, tob_errs):
        self.nobs = nobs
        self.tobs = tobs
        self.tob_errs = tob_errs

class OutburstTimeSamples:
    def __init__(self, nobs, tob_samples, bws):
        self.nobs = nobs
        self.tob_samples = tob_samples
        self.bandwidths = bws

def KDEUnivariate(samples, bw):
    kde_skl = KernelDensity(bandwidth=bw)
    kde_skl.fit(samples[:, np.newaxis])
    
    return kde_skl

def lnlike_fn(model, outburst_numbers, outburst_time_samples, bandwidths, redshift, epsabs=1e-14, epsrel=1e-14, init_step=0.1):
    
    KDEs = [KDEUnivariate(tobs_samples, bw) for tobs_samples, bw in zip(outburst_time_samples, bandwidths)]
    phis = np.pi*outburst_numbers
    
    def lnlike(params):
        tobs_model = np.array( [[ model.outburst_times_E(params, phis, redshift, epsabs, epsrel, init_step) ]] )
        
        result = np.sum([ KDE.score_samples(tob_model) for KDE, tob_model in zip(KDEs, tobs_model) ])
        
        return result
    
    return lnlike

class ModelSetup:
    
    def __init__(self, model, ndim, prior_lims, outburst_time_data, redshift):
        
        self.model = model
        self.ndim = model.N_PARAMS
        self.prior_transform = uniform_prior_transform_fn(prior_lims)
        
        self.data = outburst_time_data
        
        if isinstance(outburst_time_data, OutburstTimes):
            self.mode = "Normal"
            self.lnlike = model.Likelihood(outburst_time_data.nobs*np.pi, outburst_time_data.tobs, outburst_time_data.tob_errs, redshift) 
        elif isinstance(outburst_time_data, OutburstTimeSamples):
            self.mode = "KDE"
            self.lnlike = lnlike_fn(model, outburst_time_data.nobs, outburst_time_data.tob_samples, redshift)
        else:
            raise ValueError("Invalid outburst_time_data")
    
def read_OutburstTimes(filename):
    nobs, tobs, tob_errs = np.genfromtxt(filename).transpose()
    return OutburstTimes(nobs, tobs, tob_errs)

def read_OutburstTimeSamples(meta_file, sample_file_fmt):
    outburst_numbers, years, bandwidths = np.genfromtxt(bandwidth_file).transpose()
    sample_files = [sample_file_fmt.format(int(year)) for year in years]
    outburst_time_samples = [np.genfromtxt(sample_file) for sample_file in sample_files]
    
    return OutburstTimeSamples(outburst_numbers, outburst_time_samples, bandwidths)

