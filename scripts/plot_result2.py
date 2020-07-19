import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6  

def plot_result_spin(res_file,title):
    with open(res_file, 'rb') as resfile:
        result = pickle.load(resfile)
        
        print(res_file, result.logz)

        samples = result.samples.copy()
        weights = result.weights
        
        z = 0.306
        
        x0 = samples[:,0]
        e0 = samples[:,1]
        #u0 = samples[:,2]
        #t0 = samples[:,3]
        M  = samples[:,4]
        eta = samples[:,5]
        Xi = samples[:,6]
        
        nb0 = x0**1.5 / M
        Pb0E = (1+z)*2*np.pi/nb0
        
        samples_new = np.array([Pb0E/year, e0, M/(1e9*MSun), eta, Xi]).transpose()
        
        labels = ["\n$P_{b0}^E$ (yr)", 
          "\n$e_0$", 
          "\n$M$ ($10^9 M_{\\odot}$)",
          "\n$\\eta$ ($10^{-2}$)",
          "\n$\\Xi$"]
        
        cfig = corner.corner(samples_new, weights=weights, labels=labels, label_kwargs = {"fontsize":12}, range=[0.999999999]*5)
        plt.suptitle(title)
        
        #for ax in cfig.get_axes():
        #   ax.tick_params(axis='both', labelsize=7)
        #    ax.locator_params(axis='y', nbins=3)
        #    ax.locator_params(axis='x', nbins=4)
        #    ax.yaxis.label.set_size(8)
        
        plt.show()

def plot_result_nospin(res_file,title):
    with open(res_file, 'rb') as resfile:
        result = pickle.load(resfile)
        
        print(res_file, result.logz)

        samples = result.samples.copy()
        weights = result.weights
        
        z = 0.306
        
        x0 = samples[:,0]
        e0 = samples[:,1]
        #u0 = samples[:,2]
        #t0 = samples[:,3]
        M  = samples[:,4]
        eta = samples[:,5]
        
        nb0 = x0**1.5 / M
        Pb0E = (1+z)*2*np.pi/nb0
        
        samples_new = np.array([Pb0E/year, e0, M/(1e9*MSun), eta]).transpose()
        
        labels = ["\n$P_{b0}^E$ (yr)", 
          "\n$e_0$", 
          "\n$M$ ($10^9 M_{\\odot}$)",
          "\n$\\eta$ ($10^{-2}$)"]
        
        cfig = corner.corner(samples_new, weights=weights, labels=labels, label_kwargs = {"fontsize":12}, range=[0.999999999]*4)
        plt.suptitle(title)
        
        #for ax in cfig.get_axes():
        #   ax.tick_params(axis='both', labelsize=7)
        #    ax.locator_params(axis='y', nbins=3)
        #    ax.locator_params(axis='x', nbins=4)
        #    ax.yaxis.label.set_size(8)
        
        plt.show()

#plot_result_spin("resultA_spin_samples.dat", "Procedure A, Spinning")
#plot_result_spin("resultB_spin_samples.dat", "Procedure B, Spinning")
plot_result_nospin("resultA_nospin_samples.dat", "Procedure A, Non-Spinning")
plot_result_nospin("resultB_nospin_samples.dat", "Procedure B, Non-Spinning")

#for resfile in ["resultA_nospin_samples.dat", "resultB_nospin_samples.dat"]:
#    plot_result(resfile, units, shifts, labels)

