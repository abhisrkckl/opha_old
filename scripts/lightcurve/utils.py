import numpy as np
from lightcurve import LightCurve

def read_outburst_numbers(filename):
    data = np.genfromtxt(filename, dtype=int)
    return {yob : nob for nob, yob in data}
    
def read_priors(filename):
    data = np.genfromtxt(filename,comments='#')
    nobs = data.shape[0]
    return data.reshape(nobs, 5, 2)
    
