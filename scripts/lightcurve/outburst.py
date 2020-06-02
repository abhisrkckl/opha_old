import numpy as np
import nestle

import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt

from lightcurve_model import lnlike_fn, prior_transform_fn
from settings import *

def read_lightcurve(filename):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    
    M_norm = max(M)-M
    return t,M_norm

def read_outburst_numbers(filename):
    data = np.genfromtxt(filename)
    return {yob : nob for nob,yob in data}

def between_mask(x, a, b): 
    return np.logical_and(x>=a, x<b)

def read_priors(filename):
    data = np.genfromtxt(filename,comments='#')
    nobs = data.shape[0]
    return data.reshape(nobs, 5, 2)

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
    
    def __init__(self, nob, parent_lightcurve, time_cut, prior, template, label=None, kill=True):
        t0,t1 = time_cut
        self.lightcurve = parent_lightcurve.slice(t0,t1)
        self.label = label
        self.priors = priors
        self.template = template
        self.nob = nob

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

outburst_numbers = read_outburst_numbers(filename=config_dir+'outburst_numbers.txt')
lightcurve = LightCurve(filename=lightcurve_dir+'OJ287_Vmag.txt')
cuts = np.genfromtxt(config_dir+"lightcurve_cuts.txt")
labels = [int(cut_min) for cut_min,cut_max in cuts]
priors = read_priors(config_dir+"priors.txt")
templates = [read_template(template_dir+"oj287_templ_all.txt".format(label)) for label in labels]
outbursts = [Outburst(outburst_numbers[]lightcurve, cut, prior, template, label=label) for cut,prior,template,label in zip(cuts,priors,templates,labels)]

param_labels = ["$t_{ob}$","$\\Delta M$", "$s$", "$A$", "$\\varsigma$"]
ndim = 5
for idx,ob in enumerate(outbursts):

    print("\nAnalyzing", ob.label)

    result = nestle.sample(ob.lnlike, ob.prior_transform, ndim, npoints=300, method='multi', 
                           #callback=nestle.print_progress
             )

    means, covs = nestle.mean_and_cov(result.samples, weights=result.weights)
    stds = np.diag(covs)**0.5
    
    for mean, std, label in zip(means,stds,param_labels):
        print("{} = {} +/- {}".format(label,mean,std))

    samples_uniweight = nestle.resample_equal(result.samples, weights=result.weights)[:,0]

    corner.corner(result.samples, weights=result.weights, labels=param_labels, range=[1-1e-5]*ndim, quantiles=[0.159,0.5,0.841])
    plt.show()
    
    rangemin, rangemax = corner.quantile(result.samples[:,0], [1e-5,1-1e-5], weights=result.weights)
    
    np.savetxt("single_template/oj287_tobs_samples_1templ_{}.txt".format(ob.label), samples_uniweight)

    samples_unique = np.sort(list(set(samples_uniweight)))
    median = np.median(samples)
    mad = np.median(np.abs(samples-median))
    std = 1.4826*mad
    print("{:0.4f} +/- {:0.4f}".format(median,std))
    summary += [[year, median, std]]

    
    #kde = KDEUnivariate(samples_uniweight, )
    #pdf_kde = np.exp( kde.log_pdf(samples_uniweight) )

    #plt.plot(samples_uniweight, pdf_kde, color='r')

plt.show()
