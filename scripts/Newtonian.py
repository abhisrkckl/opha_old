import numpy as np
import matplotlib.pyplot as plt
import Newtonian_py as Newt
import Newtonian_anl as Nanl
import time

year = 365*24*3600
day = year/365

n = 2*np.pi/year
e = 0.5
w = np.pi/6
t0  = 0
params = [t0,n,e,w]

phis = np.pi*np.linspace(1,20,20)
begin_t = time.time()
outbursts_num = Newt.outburst_times(params, phis)
print time.time()-begin_t
outbursts_fake = Nanl.TimeElapsed(params, phis) + day*np.random.randn(len(phis))

plt.plot(outbursts_num/year, phis/np.pi)
plt.plot(outbursts_fake/year, phis/np.pi)
#plt.plot(phis/np.pi, (outbursts_fake-outbursts_num)/day)
plt.grid()
plt.show()

"""

t0s = year*np.linspace(-2,2,1000) 
paramss = [(t0_,n,e,w) for t0_ in t0s]
likelihood_fn = Newt.Likelihood(phis, outbursts_fake, day*np.ones_like(outbursts_fake)) 
#likelihoods = [likelihood_fn(params_) for params_ in paramss]
begin_t = time.time()
likelihoods = list(map(likelihood_fn, paramss))
print time.time()-begin_t
plt.plot(t0s/year, likelihoods)
plt.show()
"""

