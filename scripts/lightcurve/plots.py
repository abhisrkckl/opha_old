import matplotlib.pyplot as plt

def plot_full_lightcurve(lightcurve_det, lightcurve_cen):
    plt.scatter(lightcurve_det.t, lightcurve_det.M, marker='x', label="detections")
    plt.scatter(lightcurve_cen.t, lightcurve_cen.M, marker='v', label='upper limits')
    plt.xlabel("t (yr)", fontsize=14)
    plt.ylabel("$M_{ref} - M$", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.legend()
    plt.grid()
    
    plt.show()
    
def plot_outburst_lightcurves(outbursts):
    no_of_obs = len(outbursts)
    no_of_cols = 4
    no_of_rows = int(np.ceil(no_of_obs/no_of_cols))
    
    for idx, outburst in enumerate(outbursts):
        plt.subplot(no_of_rows, no_of_cols, idx+1)
        
        plt.scatter(outburst.lightcurve_det.t-outburst.year, outburst.lightcurve_det.M, marker='x')
        plt.scatter(outburst.lightcurve_cen.t-outburst.year, outburst.lightcurve_cen.M, marker='v')
        
        plt.xlabel("t(yr)-{}".format(outburst.year), fontsize=14)
        
    plt.show()
