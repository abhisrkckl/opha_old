import numpy as np

def model_fn(x0,y0):
    def templ(x):
        return np.interp(x,x0,y0)
    
    def model(x, params):
        dx,dy,s,a = params[0:4]
        x1 = (x-dx)/s

        outq = not np.any( np.logical_and( x1>=min(x0), x1<=max(x0) ) )

        return outq, dy + a*templ(x1) 
    
    return model

def lnlike_fn(x0,y0,x1,y1,kill=False):
    model = model_fn(x0,y0)
    
    def lnlike(params):
        outq, y_pred = model(x1, params[:-1])
        err = params[-1]
        return -sum((y_pred-y1)**2)/2/err**2 - len(y1)*np.log(err) if not (outq and kill) else -np.inf
    
    return lnlike    

def prior_transform_fn(params_min, params_max):
    spans = np.array(params_max)-np.array(params_min)
    
    def prior_transform(x):
	    return spans*x + np.array(params_min)
    
    return prior_transform
