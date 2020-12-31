import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm as gaussian
from sklearn.model_selection import GridSearchCV

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def kde_bandwidth_cv(samples, bwmin=0.1, bwmax=1.0, cv=20):
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(bwmin, bwmax, 100)},
                        cv=cv) # 20-fold cross-validation
    grid.fit(samples[:, None])
    return grid.best_params_['bandwidth']

def sample_mode(samples):
    bw = kde_bandwidth_cv(samples)
    pdf_samples = kde_sklearn(samples, samples, bandwidth=bw)
    maxidx = np.argmax(pdf_samples)
    return samples[maxidx]

samples = np.concatenate(( 5+0.5*np.random.randn(1000),
                           2+0.7*np.random.randn(500), ))



x_grid = np.linspace(0,7,128)
pdf_true = (2./3 * gaussian(5,0.5).pdf(x_grid) +
            1./3 * gaussian(2,0.7).pdf(x_grid))
bw_cv = kde_bandwidth_cv(samples)
pdf_kde = kde_sklearn(samples, x_grid, bandwidth=bw_cv)

mode = sample_mode(samples)

plt.hist(samples, bins=16, density=True)
plt.plot(x_grid,pdf_kde)
plt.plot(x_grid,pdf_true)
plt.axvline(mode)
plt.show()