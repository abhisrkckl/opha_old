import numpy as np
from sklearn.neighbors import KernelDensity

def KDEUnivariate(samples, bw):
    kde_skl = KernelDensity(bandwidth=bw)
    kde_skl.fit(samples[:, np.newaxis])
    
    return kde_skl

def lnlike_fn(model, outburst_numbers, outburst_time_samples, bandwidths, redshift, epsabs, epsrel, init_step):
    
    KDEs = [KDEUnivariate(tobs_samples, bw) for tobs_samples, bw in zip(outburst_time_samples, bandwidths)]
    phis = np.pi*outburst_numbers
    
    def lnlike(params):
        tobs_model = np.array( [[ model.outburst_times_E(params, phis, redshift, epsabs, epsrel, init_step) ]] )
        
        result = np.sum([ KDE.score_samples(tob_model) for KDE, tob_model in zip(KDEs, tobs_model) ])
        
        return result
    
    return lnlike

def lnlike_fn_from_file(model, bandwidth_file, sample_file_fmt, redshift, epsabs, epsrel, init_step):
    outburst_numbers, years, bandwidths = np.genfromtxt(bandwidth_file).transpose()
    
    sample_files = [sample_file_fmt.format(int(year)) for year in years]
    
    outburst_time_samples = [np.genfromtxt(sample_file) for sample_file in sample_files]
    
    return lnlike_fn(model, outburst_numbers, outburst_time_samples, bandwidths, redshift, epsabs, epsrel, init_step)
