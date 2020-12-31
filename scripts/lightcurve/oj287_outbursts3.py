import numpy as np
import matplotlib.pyplot as plt  
import scipy.optimize as sp_opt
import nestle
import corner
#from distribution import KDEUnivariate

#***** Prepare Lightcurves *************************************



t_1912,M_1912=slice_data(ts,Ms,1912.9,1913.1)
t_1934,M_1934=slice_data(ts,Ms,1934.0,1934.3)
t_1947,M_1947=slice_data(ts,Ms,1947.2,1947.325)
t_1957,M_1957=slice_data(ts,Ms,1957.05,1957.23)
t_1964,M_1964=slice_data(ts,Ms,1964.15,1964.3)
t_1972,M_1972=slice_data(ts,Ms,1972.5,1973.172)
t_1982,M_1982=slice_data(ts,Ms,1982.87,1983.09)
t_1984,M_1984=slice_data(ts,Ms,1984.05,1984.25)
t_1995,M_1995=slice_data(ts,Ms,1995.8,1995.92)
t_2005,M_2005=slice_data(ts,Ms,2005.6,2006.00)
t_2007,M_2007=slice_data(ts,Ms,2007.68,2007.745)
t_2015,M_2015=slice_data(ts,Ms,2015.80,2016.02)
t_2019,M_2019=slice_data(ts,Ms,2019.50,2019.99)

t_arr = [t_1912, 
         t_1934, 
         t_1947, 
         #t_1957, 
         #t_1964, 
         t_1972, 
         t_1982, 
         t_1984, 
         t_1995, 
         t_2005, 
         t_2007, 
         t_2015, 
         t_2019
         ]
M_arr = [M_1912, 
         M_1934, 
         M_1947, 
         #M_1957, 
         #M_1964, 
         M_1972, 
         M_1982, 
         M_1984, 
         M_1995, 
         M_2005, 
         M_2007, 
         M_2015, 
         M_2019
         ]

#***** Compute Template 0 *************************************

def bin_lightcurve(ts, Ms, nbins, tfid=0): 
    
    tmin = np.min(ts)
    tmax = np.max(ts)
    
    lo = tmin + np.arange(nbins)*(tmax-tmin)/nbins
    hi = tmin + (np.arange(nbins)+1)*(tmax-tmin)/nbins
    md = tmin + (np.arange(nbins)+0.5)*(tmax-tmin)/nbins

    Mb = np.array( [np.median(Ms[between_mask(ts,a,b)]) for a,b in zip(lo,hi) if sum(between_mask(ts,a,b))>0] )
    tb = np.array( [t for a,t,b in zip(lo,md,hi) if sum(between_mask(ts,a,b))>0] )

    return tb-tfid, Mb

def bin_lightcurve_except(tss, Mss, nbins, idx, tfid=0): 
    
    ts = sum([tsi for jdx,tsi in enumerate(tss) if jdx!=idx],[])
    Ms = sum([Msi for jdx,Msi in enumerate(Mss) if jdx!=idx],[])

    print(ts, Ms)

    tmin = np.min(ts)
    tmax = np.max(ts)
    
    lo = tmin + np.arange(nbins)*(tmax-tmin)/nbins
    hi = tmin + (np.arange(nbins)+1)*(tmax-tmin)/nbins
    md = tmin + (np.arange(nbins)+0.5)*(tmax-tmin)/nbins

    Mb = np.array( [np.median(Ms[between_mask(ts,a,b)]) for a,b in zip(lo,hi) if sum(between_mask(ts,a,b))>0] )
    tb = np.array( [t for a,t,b in zip(lo,md,hi) if sum(between_mask(ts,a,b))>0] )

    return tb-tfid, Mb

#tfid = 2015.8812893055556
#tb0, Mb0 = bin_lightcurve(t_2015,M_2015,36,tfid=tfid)

tb0, Mb0 = np.genfromtxt("oj287_templ2.txt").transpose()

#plt.plot(t_2015-tfid,M_2015)
#plt.plot(tb0, Mb0)
#plt.show()


#***** Model and Likelihood function *************************************

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

def lnlike_fn(x0,y0,x1,y1, kill):
    model = model_fn(x0,y0, kill)
        
    def lnlike(params):
        y_pred = model(x1, params[:-1])
        err = params[-1]
        return -sum((y_pred-y1)**2)/2/err**2 - len(y1)*np.log(err)
    
    return lnlike

def prior_transform_fn(params_min, params_max):
    spans = np.array(params_max)-np.array(params_min)
    def prior_transform(x):
	    return spans*x + np.array(params_min)
    return prior_transform

def filter_samples(samples,weights):
    mask = weights>0
    return samples[mask], weights[mask]

def obs_to_templ(x1,y1,params):
    dx,dy,s,A = params
    x1p = (x1-dx)/s
    y1p = (y1-dy)/A
    return x1p,y1p

param_labels = ["$t_{ob}$","$\\Delta M$", "$s$", "$A$", "$\\varsigma$"]

def align_lc(tT, MT, t, M, prior_min, prior_max, cornerplot=False):
    lnlike = lnlike_fn(tb0,Mb0, t,M, False)
    prior_transform = prior_transform_fn(prior_min, prior_max)
    ndim = len(prior_min)
    result = nestle.sample(lnlike, prior_transform, ndim, npoints=300, callback=nestle.print_progress)
    
    sample_maxlike = result.samples[ np.argmax(result.logl) ]

    if cornerplot:
        corner.corner(result.samples, weights=result.weights, range=[0.9999]*ndim, labels=param_labels, truths=sample_maxlike)
        plt.show()

    return sample_maxlike[:-1]

def plot_aligned(tT, MT, ts, Ms, paramss):
    plt.plot(tb0, Mb0, 'r')
    for t,M,params in zip(ts, Ms, paramss):
        t1, M1 = obs_to_templ(t,M,params)
        plt.scatter(t1, M1)
    plt.show()

def compare_templates(t1, M1, t2, M2):
    c1 = sum((M1 - np.interp(t1,t2,M2))**2)
    c2 = sum((M2 - np.interp(t2,t1,M1))**2)
    return ((c1+c2)/(len(M1)+len(M2)))**0.5

#***** Align a lightcurve to a template *************************************

prior_mins = ((1912.80,  0,   0.5,0.1, 0.01),
              (1934.00,  -2,  0.5,0.01, 0.01),
              (1947.16,  -4,  0.7,0.3, 0.01),
              #(1956.50,  -4,  0.3,0.01, 0.01),
              #(1964.10,  1.5, 0.5,0.1, 0.01),
              (1972.85,  0,   0.3,0.15,0.02),
              (1982.98,  1.5, 0.5,0.5, 0.05),
              (1984.14,  1.5, 1.0,0.7, 0.05),
              (1995.80,  0,   0.6,0.2, 0.04),
              (2005.65,  1.5, 2.0,0.3, 0.05),
              (2007.68,  1.5, 0.3,0.2, 0.01),
              (2015.80,  1.5, 0.5,0.7, 0.05),
              (2019.45,  0,   0.3,0.2, 0.05)
              )
prior_maxs = ((1913.05,  6,   2.0,2.5, 0.70),
              (1934.25,  6,   2.0,2.5, 1.50),
              (1947.22,  6,   2.0,2.5, 0.40),
              #(1957.50,  6,   5.0,5.0, 1.50),
              #(1964.24,  3.1, 1.0,3.5, 0.70),
              (1973.05,  6,   5.0,1.1, 0.30),
              (1983.95, 4,    1.2,1.5, 0.40),
              (1984.25, 4,    1.2,1.5, 0.40),
              (1995.86,  4,   2.0,1.1, 0.25),
              (2005.8,  4,    5.0,1.1, 0.25),
              (2007.70, 3.3,  0.7,0.7, 0.16),
              (2016.10, 4,    1.2,1.2, 0.40),
              (2019.60, 3.1,  2.0,1.5, 0.25)
              )


t_aligned = []
M_aligned = []

paramss_align = []
for idx, (ti, Mi, prior_min, prior_max) in enumerate(zip(t_arr, M_arr, prior_mins, prior_maxs)):
    
    #if idx not in [1]:
    #    continue

    params_align = align_lc(tb0, Mb0, ti, Mi, prior_min, prior_max, cornerplot=False)
    paramss_align.append(params_align)

    t1, M1 = obs_to_templ(ti,Mi,params_align)
    t_aligned.append(list(t1))
    M_aligned.append(list(M1))

    #plot_aligned(tb0, Mb0, (ti,), (Mi,), (params_align,))
t_aligned_flat = sum(t_aligned,[])
M_aligned_flat = sum(M_aligned,[])
tb1, Mb1 = bin_lightcurve(np.array(t_aligned_flat), np.array(M_aligned_flat), 33, tfid=0)

plt.scatter(t_aligned_flat, M_aligned_flat)
plt.plot(tb0, Mb0, 'r')
plt.plot(tb1, Mb1, 'g')

print('\n', compare_templates(tb0, Mb0, tb1, Mb1))

plt.show()

np.savetxt("oj287_templ_all.txt", np.array([tb1, Mb1]).transpose())

#plt.plot(tb0, Mb0)

markers = ['o','v','1','s','p','P','+','x','D','X','<']

for idx, (tsi, Msi) in enumerate(zip(t_aligned,M_aligned)):
    
    plt.subplot(3,4,idx+1)

    plt.scatter(tsi, Msi, marker=markers[idx], color='r')
    
    t_aligned_flat = sum([t_aligned_i for jdx,t_aligned_i in enumerate(t_aligned) if jdx!=idx],[])
    M_aligned_flat = sum([M_aligned_i for jdx,M_aligned_i in enumerate(M_aligned) if jdx!=idx],[])
    tb1, Mb1 = bin_lightcurve(np.array(t_aligned_flat), np.array(M_aligned_flat), 33, tfid=0)

    plt.plot(tb1, Mb1, marker=markers[idx])

    filename = "oj287_templ_" + str(int(prior_mins[idx][0])) + ".txt"
    np.savetxt(filename, np.array([tb1, Mb1]).transpose())

plt.show()

#for idx, (tsi, Msi) in enumerate(zip(t_aligned,M_aligned)):
#    tbi,Mbi = bin_lightcurve_except(t_aligned,M_aligned, 33, idx, tfid=0)
#    plt.plot(tbi,Mbi)
#    plt.scatter(t_aligned[idx],M_aligned[idx])
#    plt.show()

#np.savetxt("oj287_templ4.txt", np.array([tb1, Mb1]).transpose())

"""
prior_min_1982 = (1982.95, 0.5, 0.5, 0.5, 0.05)
prior_max_1982 = (1983.95, 4.0, 1.5, 1.5, 0.40)
params_align_1982 = align_lc(tb0, Mb0, t_1982, M_1982, prior_min_1982, prior_max_1982, cornerplot=True)

prior_min_1984 = (1984.12, 0.3, 1.0,0.7, 0.05)
prior_max_1984 = (1984.17, 1.5, 1.3,1.2, 0.40)
params_align_1984 = align_lc(tb0, Mb0, t_1984, M_1984, prior_min_1984, prior_max_1984)

plot_aligned(tb0, Mb0, 
                (t_1982,t_1984), 
                (M_1982,M_1984), 
                (params_align_1982, params_align_1984))
"""


"""
print()

tobs_mode = KDEUnivariate(samples[:,0], weights=weights).mode
print('tobs = ',tobs_mode)
dM_mode = KDEUnivariate(samples[:,1], weights=weights).mode
print('dM = ',dM_mode)
s_mode = KDEUnivariate(samples[:,2], weights=weights).mode
print('s = ',s_mode)
A_mode = KDEUnivariate(samples[:,3], weights=weights).mode
print('A = ',A_mode)

corner.corner(result.samples, weights=result.weights, range=[0.9999]*ndim, labels=param_labels, truths=[tobs_mode, dM_mode, s_mode, A_mode, None])
plt.show()
"""