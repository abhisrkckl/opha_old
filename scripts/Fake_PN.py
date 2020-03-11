import numpy as np
import BinX_PN_py as BinX_PN

year = 365*24*3600
day = year/365
MSun = 4.92703806e-6

z   = 0.306
M = 18498130000*MSun
eta = 0.008
Pb0 = 12.18/(1+z)/1.107
nb0 = 2*np.pi/(9*year)
x0  = (M*nb0)**(2./3)
e0  = 0.67
u0  = 0.1
t0  = 1886.623*year

params = [x0, e0, u0, t0, M, eta]

terr = 0

phis = np.pi*np.linspace(1,20,20)
outbursts_num = BinX_PN.outburst_times(params, phis)
outbursts_num = t0 + (outbursts_num-t0)*(1+z)

outbursts_fake = outbursts_num + terr*np.random.randn(len(phis))

fake_data = np.transpose([phis/np.pi,outbursts_fake/year,np.ones_like(phis)*terr/day])
np.savetxt("fake_data_PN_oj287.txt",fake_data)
