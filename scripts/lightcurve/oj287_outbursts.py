import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import nestle
import corner

def model_fn(x0,y0, kill):
    def templ(x):
        if kill:
            return np.interp(x,x0,y0,left=-100,right=-100)
        else:
            return np.interp(x,x0,y0)
    
    def model(x, params):
        dx,dy,s,a = params[0:4]
        return dy + a*templ((x-dx)/s) 
    
    return model

def chisq_fn(x0,y0,x1,y1):
    model = model_fn(x0,y0, False)
    
    def chisq(params):
        y_pred = model(x1, params)
        return sum((y_pred-y1)**2)
    
    return chisq

def read_lightcurve(filename):
    t,M = np.genfromtxt(filename,comments='#').transpose()
    M_norm = max(M)-M
    return t,M_norm
ts,Ms = read_lightcurve('OJ287_Vmag.txt')

def obs_to_templ(x1,y1,params):
    dx,dy,s,A = params
    x1p = (x1-dx)/s
    y1p = (y1-dy)/A
    return x1p,y1p

def slice_data(ts, Ms, t0, t1): 
    mask = np.logical_and(ts>=t0, ts<=t1) 
    return ts[mask], Ms[mask]

t_1912,M_1912=slice_data(ts,Ms,1912.9,1913.1)
t_1934,M_1934=slice_data(ts,Ms,1934.0,1934.3)
t_1947,M_1947=slice_data(ts,Ms,1947.2,1947.325)
t_1957,M_1957=slice_data(ts,Ms,1957.05,1957.23)
t_1964,M_1964=slice_data(ts,Ms,1964.15,1964.3)
t_1972,M_1972=slice_data(ts,Ms,1972.5,1973.20)
t_1982,M_1982=slice_data(ts,Ms,1982.87,1983.09)
t_1984,M_1984=slice_data(ts,Ms,1984.05,1984.25)
t_1995,M_1995=slice_data(ts,Ms,1995.8,1995.92)
t_2005,M_2005=slice_data(ts,Ms,2005.6,2005.99)
t_2007,M_2007=slice_data(ts,Ms,2007.685,2007.75)
t_2015,M_2015=slice_data(ts,Ms,2015.80,2016.0)
t_2019,M_2019=slice_data(ts,Ms,2019.50,2019.99)

"""

t_t = t_2015-min(t_2015)
M_t = M_2015-min(M_2015)

chisq = chisq_fn(t_t,M_t,t_2015,M_2015)
params_2015 = (2015.80,min(M_2015),1,1)
res = minimize(chisq, params_2015, method='nelder-mead', options={'xtol': 1e-8, 'disp': True,  'maxiter':10000})
model = model_fn(t_t,M_t)

#plt.scatter(t_2015, M_2015)
#plt.plot(t_2015, model(t_2015,res.x))
#plt.show()

##

chisq = chisq_fn(t_t,M_t,t_1984,M_1984)
params_1984 = (1984.05,min(M_1984),1,1)
res = minimize(chisq, params_1984, method='nelder-mead', options={'xtol': 1e-8, 'disp': True, 'maxiter':10000})
params_1984 = res.x
tp_1984,Mp_1984 = obs_to_templ(t_1984,M_1984,params_1984)

##

chisq = chisq_fn(t_t,M_t,t_1982,M_1982)
params_1982 = (1982.87,min(M_1982),1,1)
res = minimize(chisq, params_1982, method='nelder-mead', options={'xtol': 1e-8, 'disp': True, 'maxiter':10000})
params_1982 = res.x
tp_1982,Mp_1982 = obs_to_templ(t_1982,M_1982,params_1982)

##

chisq = chisq_fn(t_t,M_t,t_1912,M_1912)
params_1912 = (1913,min(M_1912),1,0.6)
res = minimize(chisq, params_1912, method='nelder-mead', options={'xtol': 1e-8, 'disp': True, 'maxiter':10000})
params_1912 = res.x
tp_1912,Mp_1912 = obs_to_templ(t_1912,M_1912,params_1912)

##
# Co-adding data
nbins = 36
t_concat = np.concatenate((t_t,tp_1984,tp_1982))
M_concat = np.concatenate((M_t,Mp_1984,Mp_1982))
t_b = np.zeros(nbins)
M_b = np.zeros(nbins)
for b in range(nbins):
    tmin,tmax = b*0.20/nbins, (b+1)*0.20/nbins
    t_b[b] = (tmin+tmax)/2
    mask = np.logical_and(t_concat>=tmin, t_concat<tmax)
    t_b[b] = np.mean(t_concat[mask])
    M_b[b] = np.mean(M_concat[mask])

##

template = np.array((t_b,M_b)).transpose()
np.savetxt("oj287_template.txt", template)
#t_t,M_t = template.transpose()
"""

template = np.genfromtxt("oj287_templ2.txt")
t_b,M_b = template.transpose()
#plt.plot(t_b,M_b)
#plt.show()

"""
plt.scatter(t_t,M_t, label="2015")
plt.scatter(tp_1984,Mp_1984, label="1984")
plt.scatter(tp_1982,Mp_1982, label="1982")
plt.plot(t_b,M_b, label="template", color="r")
plt.legend()
plt.show()
"""
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


def lnlike_fn(x0,y0,x1,y1, kill):
    model = model_fn(x0,y0, kill)
    
    #err = 0.125544611971501
    
    def lnlike(params):
        y_pred = model(x1, params[:-1])
        err = params[-1]
        return -sum((y_pred-y1)**2)/2/err**2 - len(y1)*np.log(err)
    
    return lnlike

#def MF_lnlike_fn(x0,y0,x1,y1):
    

def prior_transform_fn(params_min, params_max):
    spans = np.array(params_max)-np.array(params_min)
    def prior_transform(x):
	    return spans*x + np.array(params_min)
    return prior_transform

data_ts = (t_1912,t_1934,t_1947,t_1957, t_1964, t_1972,t_1982,t_1984,t_1995,t_2005,t_2007,t_2015,t_2019)
data_Ms = (M_1912,M_1934,M_1947,M_1957, M_1964, M_1972,M_1982,M_1984,M_1995,M_2005,M_2007,M_2015,M_2019)
prior_mins = ((1912.95,  2,   0.5,0.1, 0.01),
              (1934.00,  0,   0.5,0.01, 0.01),
              (1947.16,  0,   0.9,0.3, 0.01),
              (1956.50,  -4,  0.3,0.01, 0.01),
              (1964.10,  1.5, 0.5,0.1, 0.01),
              (1972.85,  0,   0.3,0.15,0.02),
              (1982.98,  1.5, 0.5,0.5, 0.05),
              (1984.14,  1.5, 1.0,0.7, 0.05),
              (1995.80,  0,   0.6,0.2, 0.04),
              (2005.65,  1.5, 2.0,0.3, 0.05),
              (2007.68,  2.4, 0.3,0.2, 0.01),
              (2015.80,  1.5, 0.5,0.7, 0.05),
              (2019.55,  1.5, 0.4,0.4, 0.05))
prior_maxs = ((1913.05,  6,   1.5,1.2, 0.50),
              (1934.25,  6,   2.0,2.5, 1.50),
              (1947.22,  6,   2.0,2.5, 0.40),
              (1957.50,  6,   5.0,5.0, 1.50),
              (1964.24,  3.1, 1.0,3.5, 0.70),
              (1973.05,  6,   5.0,1.1, 0.30),
              (1983.95, 4,    1.2,1.5, 0.40),
              (1984.25, 4,    1.2,1.2, 0.40),
              (1995.86,  4,   2.0,1.1, 0.25),
              (2005.8,  4,    5.0,1.1, 0.25),
              (2007.70, 3.3,  0.5,0.7, 0.16),
              (2016.10, 4,    1.2,1.2, 0.40),
              (2019.60, 3.1,  1.075,1.5, 0.25))
#t1s=[]
#M1s=[]
iplt = 0
for idx,(ti,Mi,prior_min,prior_max) in enumerate(zip(data_ts,data_Ms,prior_mins,prior_maxs)):
    
    if idx in [1,3,4]:
        continue
    
    kill = idx in [2,3,5]
    
    lnlike = lnlike_fn(t_b,M_b, ti,Mi, kill)
    prior_transform = prior_transform_fn(prior_min, prior_max)
    ndim = 5
    result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
    means,covs = nestle.mean_and_cov(result.samples, result.weights)
    
    idx_max = np.argmax(result.logl)
    modes = result.samples[idx_max]
    
    print("tob = %0.3f +/- %0.3f,   sf = %.1f"%(means[0], np.sqrt(covs[0,0]),means[2]))
    
    """
    plt.figure(3)
    corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A","err"], 
                  quantiles=[0.5]*ndim, 
                  truths = modes,
                  range=[0.999999999]*5   )
    plt.show()
    """
    
    #model = model_fn(t_b,M_b)
    #model_mean = model(ti,means)
    
    #dt,dM,s,A,err = means
    
    #t1 = (ti-dt)/s
    #M1 = (Mi-dM)/A
    #plt.scatter(t1,M1)
    
    #t1s += [t1]
    #M1s += [M1]
    
    plt.figure(1)
    plt.subplot(3,4,iplt+1)
    dx,dy,s,A,err = modes
    tref = int(means[0])
    plt.scatter(ti-tref,Mi,marker='x',color="green")
    plt.plot(s*t_b+dx-tref, A*M_b+dy )
    plt.axvline(dx-tref, color="red")
    plt.xlabel("t$-$"+str(tref)+" (year)", fontsize=14)
    if iplt%4==0:
        plt.ylabel("17.42$-$M'", fontsize=14)
    plt.tick_params(labelsize=12)
    #plt.grid()
    
    plt.figure(2)
    plt.scatter((ti-dx)/s, (Mi-dy)/A, label=str(int(min(ti))))
    
    iplt+=1
    
    #if idx==1:
    #    break

plt.figure(2)
plt.plot(t_b,M_b,color='black',linewidth=2, label="Template")
plt.axvline(0, color="red")
plt.tick_params(labelsize=12)
plt.xlabel("t' (year)", fontsize=14)
plt.ylabel("17.42$-$M'", fontsize=14)
plt.legend()

plt.show()

"""
t1_all = np.concatenate([ti for idx,ti in enumerate(data_ts) if idx not in [1,3,4]])
M1_all = np.concatenate([Mi for idx,Mi in enumerate(data_Ms) if idx not in [1,3,4]])
# Co-adding data
nbins = 32
t_b = np.zeros(nbins)
M_b = np.zeros(nbins)
for b in range(nbins):
    tmin,tmax = b*0.20/nbins, (b+1)*0.20/nbins
    t_b[b] = (tmin+tmax)/2
    mask = np.logical_and(t1_all>=tmin, t1_all<tmax)
    t_b[b] = np.mean(t1_all[mask])
    M_b[b] = np.mean(M1_all[mask])

plt.scatter(t1_all,M1_all)
#plt.plot(t_b,M_b)
#np.savetxt("oj287_templ2.txt",np.array((t_b,M_b)).transpose())
plt.show()
"""
"""


##

"""


"""
## 1912
lnlike = lnlike_fn(t_b,M_b, t_1912,M_1912)
prior_transform = prior_transform_fn((1912.5,0,0.5,0.1), (1913.1,10,3,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 1947
lnlike = lnlike_fn(t_b,M_b, t_1947,M_1947)
prior_transform = prior_transform_fn((1946.5,0,0.5,0.5), (1947.325,8,5,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 1957
lnlike = lnlike_fn(t_b,M_b, t_1957,M_1957)
prior_transform = prior_transform_fn((1956.5,0,0.9,0.1), (1957.25,10,1.1,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 1972
lnlike = lnlike_fn(t_b,M_b, t_1972,M_1972)
prior_transform = prior_transform_fn((1972.5,0,0.5,0.1), (1973.18,10,3,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 1982
lnlike = lnlike_fn(t_b,M_b, t_1982,M_1982)
prior_transform = prior_transform_fn((1982.8,0,0.5,0.5), (1983.09,5,1.5,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 1984
lnlike = lnlike_fn(t_b,M_b, t_1984,M_1984)
prior_transform = prior_transform_fn((1983.7,0,0.5,0.5), (1984.25,5,1.5,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 1995
lnlike = lnlike_fn(t_b,M_b, t_1995,M_1995)
prior_transform = prior_transform_fn((1995.4,0,0.5,0.5), (1996.0,5,1.5,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()

## 2005
lnlike = lnlike_fn(t_b,M_b, t_2005,M_2005)
prior_transform = prior_transform_fn((2005.5,1,0.8,0.4), (2005.9,3.2,1.6,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
plt.show()

## 2007
lnlike = lnlike_fn(t_b,M_b, t_2007,M_2007)
prior_transform = prior_transform_fn((2007.5,1,0.8,0.4), (2007.8,3.2,1.6,5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
plt.show()

## 2015
lnlike = lnlike_fn(t_b,M_b, t_2015,M_2015)
prior_transform = prior_transform_fn((2015.65,1,0.8,0.4), (2015.85,3.2,1.6,1.5))
ndim = 4
result = nestle.sample(lnlike, prior_transform, ndim, npoints=300)
means,covs = nestle.mean_and_cov(result.samples, result.weights)
#corner.corner(result.samples, weights=result.weights, labels=["tob","dM","s","A"], quantiles=[0.5]*ndim)
print("tob = %0.3f +/- %0.3f"%(means[0], np.sqrt(covs[0,0])))
#plt.show()
"""
