import numpy as np

def TimeElapsed(params, phi):
    
    (t0, n, e, omega) = params 
    
    #cycles = phi//(2*pi)
    #phi = phi%(2*pi)

    # True Anomaly
    v = phi-omega
    v0 = -omega
    
    # Eccentric anomaly
    cosv = np.cos(v)
    cosv0 = np.cos(v0)
    sinv = np.sin(v)
    sinv0 = np.sin(v0)
    cosu = (e + cosv) / (1 + e*cosv)
    cosu0 = (e + cosv0) / (1 + e*cosv0)
    sinu = np.sqrt(1-e*e) * sinv / (1 + e*cosv)
    sinu0 = np.sqrt(1-e*e) * sinv0 / (1 + e*cosv0)
    #u    = np.arctan2(sinu,cosu)
    
    beta = (1-np.sqrt(1-e*e))/e if e>0 else 0
    v_u  = 2*np.arctan2(beta*sinu,1-beta*cosu)
    v0_u0  = 2*np.arctan2(beta*sinu0,1-beta*cosu0)
    u    = v - v_u
    u0    = v0 - v0_u0
    
    # Mean anomaly
    l = u - e*sinu
    l0 = u0 - e*sinu0
    
    # Remove ambiguity of 2*pi
    #l += cycles*2*pi

    # Time elapsed
    t = t0 + (l-l0)/n

    return t
