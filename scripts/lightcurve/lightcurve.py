import numpy as np
from lightcurve_model import lnlike_fn, prior_transform_fn
from settings import *

def read_lightcurve(filename):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    M_norm = max(M)-M
    return t,M_norm

def between_mask(x, a, b): 
    return np.logical_and(x>=a, x<b)

class LightCurve:
    def __init__(self, filename=None, ts=None, Ms=None):
        if (ts is None or Ms is None) and filename is not None:
            self.t, self.M = read_lightcurve(filename)
        else:
            self.t, self.M = ts, Ms
    
    def slice(self, t0, t1): 
        mask = between_mask(self.t, t0, t1) 
        t_slice, M_slice = self.t[mask], self.M[mask]
        return LightCurve(ts=t_slice, Ms=M_slice)

def read_template(filename):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    return LightCurve(ts=t,Ms=M)

class Outburst:
    def __init__(self, year, number, parent_lightcurve, time_cut, prior, template, kill=True):
        t0,t1 = time_cut
        self.year = year
        self.number = number
        self.lightcurve = parent_lightcurve.slice(t0,t1)
        self.priors = prior
        self.template = template

        prior_mins, prior_maxs = prior.transpose()
        self.prior_transform = prior_transform_fn(prior_mins, prior_maxs)

        self.lnlike = lnlike_fn(template.t, template.M, self.lightcurve.t, self.lightcurve.M, kill)
    
    def obs_to_templ(self, params):
        x1,y1 = self.lightcurve.t, self.lightcurve.M
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


