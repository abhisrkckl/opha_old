import numpy as np
import nestle
import awkde
from lightcurve import *
from settings import *

lightcurve = LightCurve(filename=lightcurve_dir+'OJ287_Vmag.txt')

config = np.genfromtxt("{}/LCconfig.txt".format(config_dir))
years = config[:,0].astype(int)
Nobs = len(years)
numbers = config[:,1].astype(int)
cuts = config[:,2:4]
priors = config[:,4:14].reshape(Nobs,5,2)
bandwidths = config[:,14]

templates = [read_template("{}/oj287_templ_all.txt".format(template_dir)) for year in years]
outbursts = [Outburst(year, number, lightcurve, cut, prior, template) 
                for year, number, cut, prior, template 
                in zip(years, numbers, cuts, priors, templates)]

#param_labels = ["$t_{ob}$","$\\Delta M$", "$s$", "$A$", "$\\varsigma$"]
ndim = 5
summary = []
for ob,bw in zip(outbursts,bandwidths):

    print("\nAnalyzing", ob.year, '... ')

    result = nestle.sample(ob.lnlike, ob.prior_transform, ndim, npoints=300, method='multi', callback=None)

    tob_samples = nestle.resample_equal(result.samples, weights=result.weights)[:,0]
    tob_median = np.median(tob_samples)
    tob_mad = np.median(np.abs(tob_samples-tob_median))
    tob_std = 1.4826*tob_mad
    summary += [[ob.year, ob.number, tob_median, tob_std]]
    
    kde = awkde.GaussianKDE(glob_bw=bw, alpha=0.5, diag_cov=False)
    kde.fit(tob_samples[:,np.newaxis])
    unique_samples = np.sort(list(set(tob_samples)))
    a = min(unique_samples)
    b = max(unique_samples)
    pdf_xs = np.linspace((11*a-b)/10, (11*b-a)/10, len(unique_samples))
    #pdf_kde = kde.predict(unique_samples[:,np.newaxis])
    pdf_kde = kde.predict(pdf_xs[:,np.newaxis])
    lnpdf_kde = np.log(pdf_kde)
    
    np.savetxt("single_template/oj287_tobs_samples_1templ_{}.txt".format(ob.year), tob_samples)
    #np.savetxt("single_template/oj287_tobs_kde_1templ_{}.txt".format(ob.year), np.array([unique_samples, lnpdf_kde]).transpose())
    np.savetxt("single_template/oj287_tobs_kde_1templ_{}.txt".format(ob.year), np.array([pdf_xs, lnpdf_kde]).transpose())

np.savetxt("single_template/summary.txt", np.array(summary))
