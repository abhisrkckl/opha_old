import numpy as np
import corner

def make_template(outbursts, filename):
    ts, Ms = [], []
    for ob in outbursts:     
        res = ob.result
        tob = corner.quantile(res.samples[:,0], 0.5, weights=res.weights)
        dM = corner.quantile(res.samples[:,1], 0.5, weights=res.weights)
        s = corner.quantile(res.samples[:,2], 0.5, weights=res.weights)
        A = corner.quantile(res.samples[:,3], 0.5, weights=res.weights)
        lc_al = ob.obs_to_templ([tob,dM,s,A])
        
        ts.extend(lc_al.t)
        Ms.extend(lc_al.M)
        
    binmins = min(ts) + np.arange(0,32)*(max(ts)-min(ts))/32
    binmaxs = min(ts) + np.arange(1,33)*(max(ts)-min(ts))/32
    
    tmplt,tmplM = [],[]
    for mn,mx in zip(binmins,binmaxs):
        mask = np.logical_and(ts>=binmins, ts<=binmaxs)
        Mmask = np.array(Ms)[mask]
        tmplt.append((mn+mx)/2)
        tmplM.append(np.median(Mmask))
    
    np.savetxt(filename, np.transpose([tmplt,tmplM])
    

