import pickle
import corner
import matplotlib.pyplot as plt

def plot_result(res_file, units, shifts, labels):
    with open(res_file, 'rb') as resfile:
        result = pickle.load(resfile)
        
        print(res_file, result.logz)

        samples = result.samples.copy()
        weights = result.weights
        
        for idx in range(len(units)):
            samples[:,idx] /= units[idx]
            samples[:,idx] -= shifts[idx]
        
        cfig = corner.corner(samples, weights=weights, labels=labels, label_kwargs = {"fontsize":12})
        
        for ax in cfig.get_axes():
            ax.tick_params(axis='both', labelsize=7)
            ax.locator_params(axis='y', nbins=3)
            ax.locator_params(axis='x', nbins=4)
            ax.yaxis.label.set_size(8)
        
        plt.show()
        
day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6  
                   
units = [0.01, 1, 1, year, 1e9*MSun, 0.01, 1, 1, 1]
shifts = [  0, 0, 0, 1886, 0, 0, 0, 0, 0]
labels = ["\n$x_0$ ($10^{-2}$)", 
          "\n$e_0$", 
          "\n$u_0$ (rad)",
          "\n$t_0$ (yr)-1886",
          "\n$M$ ($10^9 M_{\\odot}$)",
          "\n$\\eta$ ($10^{-2}$)",
          "\n$\\Xi$",
          "\n$d_{em}$",
          "\n$d_{dd}$"]

for resfile in ["resultB_spin_samples.dat", "resultA_spin_samples.dat"]:
    plot_result(resfile, units, shifts, labels)

units = [0.01, 1, 1, year, 1e9*MSun, 0.01, 1, 1]
shifts = [  0, 0, 0, 1886, 0, 0, 0, 0, 0]
labels = ["\n$x_0$ ($10^{-2}$)", 
          "\n$e_0$", 
          "\n$u_0$ (rad)",
          "\n$t_0$ (yr)-1886",
          "\n$M$ ($10^9 M_{\\odot}$)",
          "\n$\\eta$ ($10^{-2}$)",
          "\n$d_{em}$",
          "\n$d_{dd}$"]

for resfile in ["resultA_nospin_samples.dat", "resultB_nospin_samples.dat"]:
    plot_result(resfile, units, shifts, labels)

