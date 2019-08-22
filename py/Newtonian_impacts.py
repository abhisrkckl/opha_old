from __future__ import print_function
import numpy as np
from Newtonian_py import emission_delay, impacts

day = 24*3600
year = 365*day

n = 2*np.pi/year
e = 0.5
w = np.pi/6
t0  = 0
params_true = [t0,n,e,w]

phis = np.pi*np.arange(1,21)

epsabs=epsrel = 1e-14
init_step = 0.1

impact_states = impacts(params_true, phis,   epsabs, epsrel, init_step)
emission_delays = [emission_delay(params_true, impact_state) for impact_state in impact_states]

print("phi t delay")
for phi,[t],emission_delay in zip(phis,impact_states,emission_delays):
	print(phi,t,emission_delay)
