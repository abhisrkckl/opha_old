import numpy as np
import nestle
import corner
import matplotlib.pyplot as plt
from likelihood import lnlike_fn, prior_transform_fn
from distribution import KDEUnivariate

def read_lightcurve(filename):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    
    M_norm = max(M)-M
    return t,M_norm

def between_mask(x, a, b): 
    return np.logical_and(x>=a, x<b)

def read_priors(filename):
    data = np.genfromtxt(filename,comments='#')
    nobs = data.shape[0]
    return data.reshape(nobs, 5, 2)

class LightCurve:

    def __init__(self, filename="OJ287_Vmag.txt", ts=None, Ms=None):
        if ts is None or Ms is None:
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
    
    def __init__(self, parent_lightcurve, time_cut, prior, template, label=None):
        t0,t1 = time_cut
        self.lightcurve = parent_lightcurve.slice(t0,t1)
        self.label = label
        self.priors = priors
        self.template = template

        prior_mins, prior_maxs = prior.transpose()
        self.prior_transform = prior_transform_fn(prior_mins, prior_maxs)

        self.lnlike = lnlike_fn(template.t, template.M, self.lightcurve.t, self.lightcurve.M, False)
    
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

lightcurve = LightCurve(filename='OJ287_Vmag.txt')
cuts = np.genfromtxt("lightcurve_cuts.txt")
labels = [int(cut_min) for cut_min,cut_max in cuts]
priors = read_priors("priors.txt")
templates = [read_template("oj287_templ_{}.txt".format(label)) for label in labels]
outbursts = [Outburst(lightcurve, cut, prior, template, label=label) for cut,prior,template,label in zip(cuts,priors,templates,labels)]

param_labels = ["$t_{ob}$","$\\Delta M$", "$s$", "$A$", "$\\varsigma$"]
ndim = 5
for idx,ob in enumerate(outbursts):

    #plt.plot(ob.template.t, ob.template.M, label=str(ob.label))

    
    #[1912, 1934, 1947, 1957, 1964, 1972, 1982, 1984, 1995, 2005, 2007, 2015, 2019]
    #if ob.label not in [1912,]:
    #   continue

    print("\nAnalyzing", ob.label)

    result = nestle.sample(ob.lnlike, ob.prior_transform, ndim, npoints=300, callback=nestle.print_progress, method='multi')

    params_maxlike = result.samples[ np.argmax(result.logl) ]

    #corner.corner(result.samples, weights=result.weights, labels=param_labels, truths=params_maxlike, range=[1-1e-5]*ndim)
    
    
    plt.subplot(4,4,idx+1)
    ob_model = ob.templ_to_obs(params_maxlike[:-1])
    #plt.subplot(555)
    yr = int(params_maxlike[:-1][0])
    plt.scatter(ob.lightcurve.t-yr, ob.lightcurve.M)
    plt.plot(ob_model.t-yr, ob_model.M, color='red')
    plt.xlabel("t_obs-{}".format(yr))
    
    
    #rangemin, rangemax = corner.quantile(result.samples[:,0], [1e-5,1-1e-5], weights=result.weights)
    
    samples_uniweight = nestle.resample_equal(result.samples, weights=result.weights)[:,0]

    #plt.hist(result.samples[:,0], weights=result.weights, density=True, label=str(ob.label),
    #         range=[rangemin,rangemax], bins=16)
    #plt.hist(samples_uniweight, density=True, label=str(ob.label),
    #         range=[rangemin,rangemax], bins=16)
    
    np.savetxt("oj287_tobs_samples_{}.txt".format(ob.label), samples_uniweight)

    #kde = KDEUnivariate(samples_uniweight, )
    #pdf_kde = np.exp( kde.log_pdf(samples_uniweight) )

    #plt.plot(samples_uniweight, pdf_kde, color='r')

    """
    plt.subplot(121)
    plt.scatter(ob.lightcurve.t, ob.lightcurve.M)
    plt.subplot(122)
    plt.plot(ob.template.t, ob.template.M)
    
    """
    #plt.legend()
plt.show()
