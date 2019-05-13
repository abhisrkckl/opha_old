import numpy as np
import Newtonian_anl as Nanl

year = 365*24*3600
day = year/365

n = 2*np.pi/year
e = 0.5
w = np.pi/6
t0  = 0
params = [t0,n,e,w]
   
phis = np.pi*np.linspace(1,20,20)
#outbursts_num = Newt.outburst_times(params, phis)
outbursts_fake = Nanl.TimeElapsed(params, phis) + day*np.random.randn(len(phis))
fake_data = np.transpose([phis/np.pi,outbursts_fake/year,np.ones_like(phis)])

np.savetxt("fake_data.txt",fake_data)
