import numpy as np
from lightcurve_model import lnlike_fn, prior_transform_fn
import nestle
import awkde

def read_lightcurve(filename, Mref=None):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    
    if Mref is None:
        Mref = max(M)
    
    M_norm = Mref - M
    
    return t, M_norm, Mref

def between_mask(x, a, b): 
    return np.logical_and(x>=a, x<b)

def read_template(filename):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    return LightCurve(ts=t,Ms=M)

class LightCurve:
    def __init__(self, filename=None, ts=None, Ms=None, Mref=None):
        if (ts is None or Ms is None) and filename is not None:
            self.t, self.M, self.Mref = read_lightcurve(filename, Mref=Mref)
        else:
            self.t, self.M = ts, Ms
            self.Mref = Mref
    
    def slice(self, t0, t1): 
        mask = between_mask(self.t, t0, t1) 
        t_slice, M_slice = self.t[mask], self.M[mask]
        return LightCurve(ts=t_slice, Ms=M_slice, Mref=self.Mref)

class Outburst:
    
    def __init__(self, nob, parent_lc_det, parent_lc_cen, time_cut, prior, template, label=None, kill=True):
        t0,t1 = time_cut
        
        self.nob = int(nob)
        self.year = int(t0)
        
        self.lightcurve_det = parent_lc_det.slice(t0,t1)
        self.lightcurve_cen = parent_lc_cen.slice(t0,t1)
        self.template = template
        
        self.label = label
        
        #self.priors = priors
        prior_mins, prior_maxs = prior.transpose()
        self.prior_transform = prior_transform_fn(prior_mins, prior_maxs)

        self.lnlike = lnlike_fn(self.template, self.lightcurve_det, self.lightcurve_cen, kill)
    
    def obs_to_templ(self, params):
        x1,y1 = self.lightcurve_det.t, self.lightcurve_det.M
        dx,dy,s,A = params
        x1p = (x1-dx)/s
        y1p = (y1-dy)/A
        return LightCurve(ts=x1p, Ms=y1p)
    
    def templ_to_obs(self, params):
        x1,y1 = self.template.t, self.template.M
        dx,dy,s,A = params
        x1p = x1*s + dx
        y1p = y1*A + dy
        return LightCurve(ts=x1p, Ms=y1p)
        
    def run_sampler(self):
        print("Running sampler for year {}".format(self.year))
        self.result = nestle.sample(self.lnlike, self.prior_transform, ndim=5, method='multi', npoints=300, dlogz=0.25)
        
    def process_samples(self, bandwidth):
        
        tob_samples = nestle.resample_equal(self.result.samples[:,0], weights=self.result.weights)
        
        self.median = np.median(tob_samples)
        self.nmad = 1.4826*np.median(np.abs(tob_samples-self.median))
        
        self.kde = awkde.GaussianKDE(glob_bw=bandwidth, alpha=0.5, diag_cov=False)
        self.kde.fit(tob_samples[:,np.newaxis])
        
        self.kde_grid = np.linspace((11*min(tob_samples)-max(tob_samples))/10, (11*max(tob_samples)-min(tob_samples))/10, len(tob_samples))
        self.kde_vals = self.kde.predict(self.kde_grid[:,np.newaxis])
    

