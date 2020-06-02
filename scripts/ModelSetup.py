import numpy as np
import importlib
import pygsl
pygsl.import_all()
from pygsl.interpolation import cspline

def get_model(model_name):
    model_pkg = model_name+'_py'
    model = importlib.import_module(model_pkg)
    return model

def get_data(data_file):
    data = np.genfromtxt(data_file, comments='#')
    data_x = np.pi*np.round(data[:,0])
    data_y = data[:,1]*year
    data_yerr = data[:,2]*year
    return data_x, data_y, data_yerr

def InterpKDE(mean, std, filename):
    yrs = 365.25*24*3600
    
    samples, lnpdf = np.genfromtxt(filename).transpose()
    kde_spline = cspline(len(samples))
    kde_spline.init(samples*yrs, lnpdf-np.log(yrs))
    tmin, tmax = min(samples)*yrs, max(samples)*yrs
    
    mean *= yrs
    std *= yrs
    
    def lnpdf(tob):
        if(tob<tmin or tob>tmax):
            return -0.5*((tob-mean)/std)**2 - 0.5*np.log(2*np.pi * std**2)
        else:
            return kde_spline.eval(tob)

    return lnpdf

""" 
def lnlike_fn(model, z, phiobs, tobs=None, tob_errs=None, interp_kdes=None):
    if interp_kdes is not None:
        def lnlike(params):
            tobs = model.outburst_times_E(params, phiobs, z)
            return sum([pdf(tob) for pdf,tob in zip(interp_kdes,tobs)])
        return lnlike 
    elif tobs is not None and tob_errs is not None:
        return model.Likelihood(phiobs, tobs, tob_errs, z)
 """

def lnlike_from_data(model, z, datafile):
    phiobs, tobs, tob_errs = get_data(datafile)
    return model.Likelihood(phiobs, tobs, tob_errs, z)

def lnlike_from_kde(model, z, kde_dir, kde_fmt):
    summary = np.genfromtxt(kde_dir+"/summary.txt")
    interp_kdes = [InterpKDE(mean, std, kde_fmt.format(kde_dir, int(year))) for year,nob,mean,std in summary]
    phiobs = np.pi*summary[:,1]
    def lnlike(params):
        tobs = model.outburst_times_E(params, phiobs, z)
        return sum([pdf(tob) for pdf,tob in zip(interp_kdes,tobs)])
    return lnlike

def lnlike_fn(model, z, datafile=None, kde_dir=None, kde_fmt=None):
    if datafile is not None:
        return lnlike_from_data(model, z, datafile)
    elif kde_dir is not None and kde_fmt is not None:
        return lnlike_from_kde(model, z, kde_dir, kde_fmt)

def prior_transform_fn(datafile):
    units, prmin, prmax = np.genfromtxt(datafile)
    prmin *= units
    prmax *= units
    spans = prmax-prmin
    def prior_transform(cube):
        return prmin + spans*cube

    return prior_transform

