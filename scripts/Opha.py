import importlib

day = 24*3600
year = 365.25*day
MSun = 4.92703806e-6

def model(model_name):
    model_pkg = model_name+'_py'
    model = importlib.import_module(model_pkg)
    return model

Spinning = model("Spin")
NonSpinning = model("NoSpin")

def read_outburst_times(tobs_file):
    data = np.genfromtxt(data_file, comments='#')
    data_x = np.pi*np.round(data[:,0])
    data_y = data[:,1]*year
    data_yerr = data[:,2]*year
    return data_x, data_y, data_yerr

def likelihood_fn(model, redshift, tobs_file, mode='tobs'):
    phiobs, tobs, toberrs = read_outburst_times(tobs_file)
    return model.Likelihood(phiobs, tobs, toberrs, redshift)

def prior_transform_fn(model, priors_file):
    
    mins, maxs = np.genfromtxt(priors_file, comments='#')
    spans = maxs-mins

    ndim = model.N_PARAMS
    if len(spans) != ndim:
        raise ValueError("Invalid number of parameters in prior file.")

    def prior_transform(x):
        return spans*x + mins

    return prior_transform
