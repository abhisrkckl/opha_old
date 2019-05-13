import numpy as np
import nestle
import corner
import matplotlib.pyplot as plt

from Newtonian_anl import TimeElapsed
from Newtonian_py import Likelihood

day = 24*3600
year = 365*day

n = 2*np.pi/year
e = 0.5
w = np.pi/6
t0  = 0
params_true = [t0,n,e,w]

data = np.genfromtxt('fake_data.txt')
data_x = np.pi*np.round(data[:,0])
data_y = data[:,1]*year
data_yerr = data[:,2]*0.1*day

anl = True

# Define a likelihood function
if anl:
	ln2pi = 1.8378770664093453
	def loglike(theta):
	    #y = line(data_x,theta)
	    y = TimeElapsed(theta, data_x)
	    chisq = np.sum(((data_y - y) / data_yerr)**2 + 2*np.log(data_yerr) )
	    return -chisq/2. - 0.5*ln2pi*len(data_x)
else:
	loglike = Likelihood(data_x, data_y, data_yerr)

# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
t0min,t0max = -day/2.,day/2.
nmin,nmax = n*0.99998, n*1.00002
emin,emax = 0.2,0.9
wmin,wmax = 0,np.pi/2
mins = np.array((t0min,nmin,emin,wmin))
maxs = np.array((t0max,nmax,emax,wmax))
spans = maxs-mins
def prior_transform(x):
    return spans*x + mins

# Run nested sampling.
ndim = len(params_true)
result = nestle.sample(loglike, prior_transform, ndim)

print "log z = ",result.logz     # log evidence
print "log z err = ",result.logzerr  # numerical (sampling) error on logz

result.samples[:,0]/=day
result.samples[:,1]*=year/(2*np.pi)
corner.corner(result.samples, weights=result.weights, 
		quantiles=[0.0455, 0.5, 0.9545], bins=15,
		labels=['t0 (day)','Fb (yr^-1)','e','w'],
		show_titles=True)
plt.show()

