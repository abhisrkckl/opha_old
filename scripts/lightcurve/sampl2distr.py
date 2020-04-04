import numpy as np
import awkde
from settings import *

years, bandwidths = np.genfromtxt(config_dir + "bandwidths.txt").transpose()

distr_type = "single_template"

filenames = ["{}/oj287_tobs_samples_1templ_{}.txt".format(distr_type,int(year)) for year in years]
sample_arrs = [np.genfromtxt(filename) for filename in filenames]

summary = []

for idx, (year, samples, bandwidth) in enumerate(zip(years, sample_arrs, bandwidths)):
    
    unique_samples = np.sort(list(set(samples)))

    median = np.median(samples)
    mad = np.median(np.abs(samples-median))
    std = 1.4826*mad
    print("{:0.4f} +/- {:0.4f}".format(median,std))
    summary += [[year, median, std]]

    kde = awkde.GaussianKDE(glob_bw=bandwidth, alpha=0.5, diag_cov=False)
    kde.fit(samples[:,np.newaxis])
    pdf_kde = kde.predict(unique_samples[:,np.newaxis])
    lnpdf_kde = np.log(pdf_kde)

    np.savetxt("{}/oj287_tobs_kde_1templ_{}.txt".format(distr_type,int(year)), 
                np.array([unique_samples, lnpdf_kde]).transpose())

np.savetxt("{}/summary.txt".format(distr_type), np.array(summary))

