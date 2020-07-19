import numpy as np
import matplotlib.pyplot as plt
import ModelSetup
import nestle
import corner
import pickle

 
model, prior_transform, lnlike_gauss, lnlike_kde = ModelSetup.setup_model('Spin', '../data/config/outburst_numbers.txt', '../data/config/spin_priors.txt', 0.306, 'A', 'gauss')
resultA_spin = nestle.sample(lnlike_kde, prior_transform, model.N_PARAMS, npoints=100, method='multi', callback=nestle.print_progress)
with open('resultA_spin_samples.dat','wb') as outfile:
    pickle.dump(resultA_spin, outfile)

#model, prior_transform, lnlike_gauss, lnlike_kde = ModelSetup.setup_model('Spin', '../data/config/outburst_numbers.txt', '../data/config/spin_priors.txt', 0.306, 'B', 'gauss')
#resultB_spin = nestle.sample(lnlike_kde, prior_transform, model.N_PARAMS, npoints=100, method='multi', callback=nestle.print_progress)
#with open('resultB_spin_samples.dat','wb') as outfile:
#    pickle.dump(resultB_spin, outfile)


#model, prior_transform, lnlike_gauss, lnlike_kde = ModelSetup.setup_model('NoSpin', '../data/config/outburst_numbers.txt', '../data/config/nospin_priors.txt', 0.306, 'A', 'gauss')
#resultA_nospin = nestle.sample(lnlike_kde, prior_transform, model.N_PARAMS, npoints=100, method='multi', callback=nestle.print_progress)
#with open('resultA_nospin_samples.dat','wb') as outfile:
#    pickle.dump(resultA_nospin, outfile)

#model, prior_transform, lnlike_gauss, lnlike_kde = ModelSetup.setup_model('NoSpin', '../data/config/outburst_numbers.txt', '../data/config/nospin_priors.txt', 0.306, 'B', 'gauss')
#resultB_nospin = nestle.sample(lnlike_kde, prior_transform, model.N_PARAMS, npoints=100, method='multi', callback=nestle.print_progress)
#with open('resultB_nospin_samples.dat','wb') as outfile:
#    pickle.dump(resultB_nospin, outfile)

