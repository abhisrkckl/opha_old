import numpy as np
import nestle
import corner
import matplotlib.pyplot as plt
from time import time
import importlib
from pylatexenc.latex2text import LatexNodes2Text

latex = LatexNodes2Text()

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

def get_model(model_name):
    model_pkg = model_name+'_py'
    model = importlib.import_module(model_pkg)
    return model

def get_data(data_file="OJ287_real.txt"):
    data = np.genfromtxt(data_file, comments='#')
    data_x = np.pi*np.round(data[:,0])
    data_y = data[:,1]*year
    data_yerr = data[:,2]*year
    return data_x, data_y, data_yerr

def make_plot_label(param_name, unit_frac, unit_name, shift):
    label = "\n"+param_name
    if unit_frac!=1:
        label += " (%s)"%unit_name
    if shift<0:
        label += "$+%0.0f$"%(-shift)
    elif shift>0:
        label += "$-%0.0f$"%(shift)

    return label

#def make_print_param(mean, std, param_name, unit_frac, unit_name):
#    label = param_name + " = " + str(mean) + " +/- " + str(std) 

def run_sampler(model, prior_transform, data, z=0.306, npts=100):

    print model.description()

    data_x, data_y, data_yerr = data

    Likelihood = model.Likelihood
    loglike = Likelihood(data_x, data_y, data_yerr, z) 

    ndim = model.N_PARAMS
    
    begin_time=time()
    result = nestle.sample(loglike, prior_transform, ndim, npoints=npts, method='multi', callback=nestle.print_progress)
    end_time=time()
    run_time =  end_time-begin_time
    print "Time elapsed = ",run_time
    print "Time per likelihood call = ",run_time/result.ncall
    

    print "log z = ",result.logz     # log evidence
    print "log z err = ",result.logzerr  # numerical (sampling) error on logz

    return result

def plot_posterior(model, data, result, display_params, save_prefix, nbins=15, z=0.306):

    ndim = model.N_PARAMS    
    
    samples = result.samples.copy()
    for iparam in range(ndim):
        unit_frac = display_params[iparam][1]
        shift = display_params[iparam][3]
        samples[:,iparam]/=unit_frac
        samples[:,iparam]-=shift

    plot_labels = [make_plot_label(*display_param) for display_param in display_params]

    #print plot_labels
    #means, cov = nestle.mean_and_cov(result.samples, result.weights)
    #print "Estimated parameters :"
    #for ipar, mean in enumerate(means):
    #    print latex.latex_to_text(plot_labels[ipar]), "=", mean, "+/-", cov[ipar,ipar]

    corner.corner(samples, weights=result.weights,
                  quantiles=[0.0455, 0.5, 0.9545], 
                  bins=nbins,
                  labels=plot_labels,
                  label_kwargs = {"labelpad":100, "fontsize":12},
                  #show_titles=True,
                  range=[0.99999999999999]*ndim,
                  use_math_text=True,
                  title_fmt="0.3f")
    
    #mean, cov = nestle.mean_and_cov(result.samples, result.weights)
    idx_t0 = model.N_STATE_PARAMS-1
    #t0 = mean[idx_t0]
    data_x,data_y,data_yerr = data
    #outburst_time_mean = model.outburst_times(mean, data_x, 1e-14, 1e-14, 0.1)
    #outburst_time_mean_z = t0 + (outburst_time_mean-t0)*(1+z)
    #outburst_time_samples_yr = np.zeros_like(outburst_time_samples)
    #chisq = sum(((data_y-outburst_time_mean_z)/data_yerr)**2) / (len(data_x)-len(mean))
    
    ##################
        
    # Inset showing timing residuals.
    
    samples_new = nestle.resample_equal(result.samples,result.weights)
    outburst_time_samples = model.outburst_times_x(samples_new, data[0], 1e-14, 1e-14, 0.1)
    t0s = samples_new[:,idx_t0]
    outburst_time_samples_yr = np.zeros_like(outburst_time_samples)
    for idx,(tob_sample, t0) in enumerate(zip(outburst_time_samples,t0s)):
        outburst_time_samples_yr[idx] = (t0 + (tob_sample-t0)*(1+z))/year
    tob_pred_means = np.mean(outburst_time_samples_yr, axis=0)
    tob_pred_stds = np.std(outburst_time_samples_yr, axis=0)
    
    chisq=0
    plt.subplot(4,3,3)
    #plt.errorbar(data_y/year, (data_y-outburst_time_mean_z)/day, data_yerr/day, fmt='+', label="$\\chi^2/\\nu = %f$"%(chisq))
    plt.errorbar(data_y/year, np.zeros_like(data_y), tob_pred_stds*year/day, label="$\\chi^2/\\nu = %f$"%(chisq), elinewidth=5, fmt=".")
    plt.errorbar(data_y/year, (data_y-tob_pred_means*year)/day, data_yerr/day, fmt='+', label="$\\chi^2/\\nu = %f$"%(chisq), elinewidth=2.5)
    plt.xlabel("$t_{ob}$ (yr)")
    plt.ylabel("Residuals (day)")
    #plt.legend()
    plt.grid()
    
    
    outburst_time_samples_all = model.outburst_times_x(samples_new, np.pi*np.arange(5,25), 1e-14, 1e-14, 0.1)
    outburst_time_samples_yr_all = np.zeros_like(outburst_time_samples_all)
    for idx,(tob_sample, t0) in enumerate(zip(outburst_time_samples_all,t0s)):
        outburst_time_samples_yr_all[idx] = (t0 + (tob_sample-t0)*(1+z))/year
    tob_pred_means_all = np.mean(outburst_time_samples_yr_all, axis=0)
    tob_pred_stds_all = np.std(outburst_time_samples_yr_all, axis=0)
    print tob_pred_means_all

    """
    mean, cov = nestle.mean_and_cov(samples, result.weights)
    std = np.diag(cov)
    plt.subplot(3,3,6)
    params_text = ""
    for m,s, label in zip(mean,std,plot_labels):
        params_text += "%s = %0.6f $\\pm$ %0.6f\n"%(label,m,s)
    plt.text(0.1,0.1,params_text)
    plt.axis('off')
    """

    plt.savefig(save_prefix+"_post.pdf")
    #plt.show()

def print_results(result, display_params):
    means, cov = nestle.mean_and_cov(result.samples, result.weights)
    std = np.sqrt(np.diag(cov))
    
    plot_labels = [make_plot_label(*display_param) for display_param in display_params]
    unit_fracs = [dp[1] for dp in display_params]
    shifts = [dp[3] for dp in display_params]
    
    print "\nEstimated parameters :"
    for m,s,label,scale,shift in zip(means,std,plot_labels,unit_fracs,shifts):
        m1 = m/scale - shift
        s1 = s/scale
        lbl = latex.latex_to_text(label)
        print "%s = %0.6f +/- %0.6f"%(lbl,m1,s1)
        

