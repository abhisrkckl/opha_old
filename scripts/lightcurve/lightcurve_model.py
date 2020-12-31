from __future__ import division
import numpy as np
from scipy.special import log_ndtr

def model_fn(x0,y0):
    def templ(x):
        return np.interp(x,x0,y0)
    
    def model(x, params):
        dx,dy,s,a = params[0:4]
        x1 = (x-dx)/s

        outq = not np.any( np.logical_and( x1>=min(x0), x1<=max(x0) ) )

        return outq, dy + a*templ(x1) 
    
    return model

""" 
def lnlike_fn(x0,y0,x1,y1,kill=False):
    model = model_fn(x0,y0)
    
    def lnlike(params):
        outq, y_pred = model(x1, params[:-1])
        err = params[-1]
        return -sum((y_pred-y1)**2)/2/err**2 - len(y1)*np.log(err) if not (outq and kill) else -np.inf
    
    return lnlike    
 """

def lnlike_fn(template, lightcurve_det, lightcurve_cen, kill=False):
    model = model_fn(template.t, template.M)
    
    def lnlike(params):
        outq_det, y_pred_det = model(lightcurve_det.t, params[:-1])
        outq_cen, y_pred_cen = model(lightcurve_cen.t, params[:-1])
        
        err = params[-1]
        
        res_det = (lightcurve_det.M - y_pred_det)/err
        res_cen = (lightcurve_cen.M - y_pred_cen)/err
        
        if outq_det and outq_cen and kill:
            return -np.inf
        else:
            lnlike_det = -sum(res_det**2)/2 - len(res_det)*np.log(err)
            lnlike_cen = sum(log_ndtr(res_cen))
            
            return lnlike_det + lnlike_cen
    
    return lnlike
 
def prior_transform_fn(params_min, params_max):
    spans = np.array(params_max)-np.array(params_min)
    
    def prior_transform(x):
	    return spans*x + np.array(params_min)
    
    return prior_transform
