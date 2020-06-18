import numpy as np
import importlib
import pygsl
pygsl.import_all()
from pygsl.interpolation import cspline
import scipy.stats

year = 365.25*25*3600

#outburst_number_file = "../data/config/outburst_numbers.txt"
#outburst_numbers, outburst_years = np.genfromtxt(outburst_number_file, dtype=int).transpose()

def get_model(model_name):
    model_pkg = model_name+'_py'
    model = importlib.import_module(model_pkg)
    return model

def get_tob_distr(outburst_number, outburst_year, procedure):
    if procedure == 'A':
        dirname = "single_template"
        filesfx = "1templ"
    elif procedure == 'A':
        procname = "round_robin"
        filesfx = "rndrob"
    else:
        raise ValueError()
        
    kde_file = "lightcurve/{}/oj287_tobs_kde_{}_{:d}.txt".format(dirname, filesfx, outburst_year)
    smpl_file = "lightcurve/{}/oj287_tobs_samples_{}_{:d}.txt".format(dirname, filesfx, outburst_year)
    
    return OutburstTimeDistribution(outburst_number, outburst_year, procedure, kde_file, smpl_file)

class OutburstTimeDistribution:
    def __init__(self, outburst_number, outburst_year, procedure, kde_file, smpl_file):
        self.nob = outburst_number
        self.year = outburst_year
        self.proc = procedure
        
        self.samples = np.genfromtxt(smpl_file) * year
        self.med = np.median(self.samples)
        self.std = 1.4826*scipy.stats.median_absolute_deviation(self.samples)
        
        self.kde_pts, self.kde_vals = np.genfromtxt(kde_file).transpose()
        self.kde_pts *= year
        self.kde_vals -= np.log(year)
        self.kde_pt_min = min(self.kde_pts)
        self.kde_pt_max = max(self.kde_pts)
        self.kde_spline = cspline(len(self.kde_pts))
        self.kde_spline.init(self.kde_pts, self.kde_vals)
        
    def gauss_lnlike(self, tob):
        return -0.5*np.log(2*np.pi*self.std**2) -0.5*((tob-self.med)/self.std)**2

    def kde_lnlike(self,tob):
        if tob<self.kde_pt_min or tob>self.kde_pt_max:
            return self.gauss_lnlike(tob)
        else:
            return self.kde_spline.eval(tob)

def prior_transform_fn(datafile, nparams):
    units, prmin, prmax = np.genfromtxt(datafile)
    prmin *= units
    prmax *= units
    spans = prmax-prmin
    def prior_transform(cube):
        return prmin + spans*cube

    if len(prmin) != nparams:
        raise ValueError()
    
    return prior_transform

class ModelSetup:
    def __init__(self, model_name, outburst_data_file, prior_file, redshift, procedure, mode):
        self.model = get_model(model_name)
        self.nparams = self.model.N_PARAMS
        
        self.nobs, self.years = np.genfromtxt(outburst_data_file, dtype=int).transpose()
        self.phiobs = np.pi*self.nobs
        self.z = redshift
        
        if mode not in ['gauss','kde']:
            raise ValueError()
        self.proc = procedure
        
        self.tob_distrs = [get_tob_distr(nob, year, self.proc) for nob, year in zip(self.nobs, self.years)]
        
        self.prior_transform = prior_transform_fn(prior_file, self.nparams)
        
    def lnlike(self, params):
        tobs_model = self.model.outburst_times_E(params, self.phiobs, self.z)
        
        if self.mode=='gauss':
            return np.sum([self.distr.gauss_lnlike(tob) for distr, tob in zip(self.tob_distrs, tobs_model)])
        elif self.mode=='kde':
            return np.sum([self.distr.kde_lnlike(tob) for distr, tob in zip(self.tob_distrs, tobs_model)])
    

