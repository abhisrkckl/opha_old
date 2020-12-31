import numpy as np
import matplotlib.pyplot as plt
import corner
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm as gaussian
from sklearn.model_selection import GridSearchCV

def kde_bandwidth_cv(samples, bwmin=0.1, bwmax=1.0, cv=20):
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(bwmin, bwmax, 30)},
                        cv=cv) # 20-fold cross-validation
    grid.fit(samples[:, None])
    return grid.best_params_['bandwidth']

class KDEUnivariate:

    def __init__(self, samples, weights=None):
        self.samples = samples
        self.weights = weights

        self.bw = kde_bandwidth_cv(samples)
        self.kde = KernelDensity(bandwidth=self.bw)
        self.kde.fit(samples[:, np.newaxis], sample_weight=weights)

        self.mean = np.average(samples, weights=weights)
        self.median = corner.quantile(samples, [0.5], weights=weights)[0]
        
        lnpdf_samples = self.log_pdf(samples)
        modeidx = np.argmax(lnpdf_samples)
        self.mode = samples[modeidx]

    def log_pdf(self,x):
        return self.kde.score_samples(x[:, np.newaxis])

"""
class KDEMultivariate:
    def __init__(self, samples, weights=None, scores=None):
        self.samples = samples
        self.weights = weights

        self.bw = kde_bandwidth_cv(samples)
        self.kde = KernelDensity(bandwidth=self.bw)
        self.kde.fit(samples, sample_weight=weights)

        self.mean = np.average(samples, weights=weights)
        self.median = corner.quantile(samples, [0.5], weights=weights)[0]
        
        if scores is None:
            scores = self.log_pdf(samples)
        modeidx = np.argmax(scores)
        self.mode = samples[modeidx]
"""

if __name__ == '__main__':
    samples = np.concatenate(( 5+0.5*np.random.randn(1000),
                            2+0.7*np.random.randn(500), ))

    x_grid = np.linspace(0,7,128)
    pdf_true = (2./3 * gaussian(5,0.5).pdf(x_grid) +
                1./3 * gaussian(2,0.7).pdf(x_grid))

    kde = KDEUnivariate(samples)
    pdf_kde = np.exp( kde.log_pdf(x_grid) )

    plt.hist(samples, density=True, bins=32)
    plt.plot(x_grid, pdf_true)
    plt.plot(x_grid, pdf_kde)
    plt.axvline(kde.mode)
    plt.show()
