import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neighbors import KernelDensity
import awkde

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

years, bandwidths = np.genfromtxt("../../data/config/bandwidths.txt").transpose()

filenames = ["single_template/oj287_tobs_samples_1templ_{}.txt".format(int(year)) for year in years]
sample_arrs = [np.genfromtxt(filename) for filename in filenames]

for idx, (year, samples, bandwidth) in enumerate(zip(years, sample_arrs, bandwidths)):
    
    samples_d = samples-year
    
    #mean = np.mean(samples)
    #err = np.std(samples)
    median = np.median(samples_d)
    mad = np.median(np.abs(samples_d-median))
    std = 1.4826*mad
    print("{:0.4f} +/- {:0.4f}".format(median+year,std))

    plt.subplot(4,4,idx+1)
    plt.hist(samples_d, density=True, bins=64, label="Histogram ({})".format(int(year)))

    grid = np.linspace(2*min(samples_d)-max(samples_d), 2*max(samples_d)-min(samples_d), 1000)
    #grid = np.linspace(min(samples_d), max(samples_d), 1000)
    kde = awkde.GaussianKDE(glob_bw=bandwidth, alpha=0.5, diag_cov=False)
    kde.fit(samples_d[:,np.newaxis])
    pdf_kde = kde.predict(grid[:,np.newaxis])
    #kde_sklearn(samples_d, grid, bandwidth)
    
    pdf_gauss = 1/np.sqrt(2*np.pi)/std * np.exp(-0.5*((grid-median)/std)**2)

    plt.plot(grid, pdf_kde, color='r', label="KDE ({})".format(int(year)))
    plt.plot(grid, pdf_gauss, color='orange', label="Gauss ({})".format(int(year)))
    plt.yscale('log')

    plt.xlabel("tobs - {}".format(int(year)))

    #plt.legend()

plt.show()

