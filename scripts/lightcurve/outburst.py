import numpy as np
import nestle
import corner
import matplotlib.pyplot as plt

#from lightcurve_model import lnlike_fn, prior_transform_fn
from settings import lightcurve_dir, config_dir, template_dir
from lightcurve import LightCurve, Outburst, read_template
from utils import read_outburst_numbers, read_priors

outburst_numbers = read_outburst_numbers(filename=config_dir+'outburst_numbers.txt')
lightcurve = LightCurve(filename=lightcurve_dir+'OJ287_Vmag.txt')
cuts = np.genfromtxt(config_dir+"lightcurve_cuts.txt")
labels = [int(cut_min) for cut_min,cut_max in cuts]
priors = read_priors(config_dir+"priors.txt")
templates = [read_template(template_dir+"oj287_templ_all.txt".format(label)) for label in labels]
outbursts = [Outburst(outburst_numbers,lightcurve, cut, prior, template, label=label) for cut,prior,template,label in zip(cuts,priors,templates,labels)]

param_labels = ["$t_{ob}$","$\\Delta M$", "$s$", "$A$", "$\\varsigma$"]
ndim = 5
for idx,ob in enumerate(outbursts):

    print("\nAnalyzing", ob.label)

    result = nestle.sample(ob.lnlike, ob.prior_transform, ndim, npoints=300, method='multi', 
                           #callback=nestle.print_progress
             )

    means, covs = nestle.mean_and_cov(result.samples, weights=result.weights)
    stds = np.diag(covs)**0.5
    
    for mean, std, label in zip(means,stds,param_labels):
        print("{} = {} +/- {}".format(label,mean,std))

    samples_uniweight = nestle.resample_equal(result.samples, weights=result.weights)[:,0]

    corner.corner(result.samples, weights=result.weights, labels=param_labels, range=[1-1e-5]*ndim, quantiles=[0.159,0.5,0.841])
    plt.show()
    
    rangemin, rangemax = corner.quantile(result.samples[:,0], [1e-5,1-1e-5], weights=result.weights)
    
    np.savetxt("single_template/oj287_tobs_samples_1templ_{}.txt".format(ob.label), samples_uniweight)

    samples_unique = np.sort(list(set(samples_uniweight)))
    median = np.median(samples)
    mad = np.median(np.abs(samples-median))
    std = 1.4826*mad
    print("{:0.4f} +/- {:0.4f}".format(median,std))
    summary += [[year, median, std]]

    
    #kde = KDEUnivariate(samples_uniweight, )
    #pdf_kde = np.exp( kde.log_pdf(samples_uniweight) )

    #plt.plot(samples_uniweight, pdf_kde, color='r')

plt.show()
