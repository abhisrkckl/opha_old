#ifndef _Opha_Likelihood_hpp_
#define _Opha_Likelihood_hpp_

#include <cmath>

namespace Opha {

    using vector_t = std::vector<double>;

    template <typename ModelClass>
    struct Likelihood {
        
        const vector_t phis, ts_outburst, terrs_outburst;
        const double z;
        const double epsabs, epsrel, init_step;
        
        Likelihood(const vector_t& _phis, const vector_t& _ts_outburst, const vector_t& _terrs_outburst, 
                   const double _z)
            : phis(_phis), ts_outburst(_ts_outburst), terrs_outburst(_terrs_outburst), 
              z(_z),
              epsabs(1e-14), epsrel(1e-14), init_step(1.) {    }
        
        Likelihood(const vector_t& _phis, const vector_t& _ts_outburst, const vector_t& _terrs_outburst, 
                   const double _z,
                   const double _epsabs, const double _epsrel, const double _init_step)
            : phis(_phis), ts_outburst(_ts_outburst), terrs_outburst(_terrs_outburst), 
              z(_z),
              epsabs(_epsabs), epsrel(_epsrel), init_step(_init_step) {    }
        
        double operator()(const typename ModelClass::params_t& params) const;
    };

    template <typename ModelClass>
    double Likelihood<ModelClass>::operator()(const typename ModelClass::params_t& params) const{

        const vector_t ts_outburst_model_E = ModelClass::outburst_times_E(params, phis, z, {epsabs, epsrel, init_step});
        
        const unsigned length = phis.size();
        double result = 0;
        for(unsigned i=0; i<length; i++){
            
            const double  diff = ts_outburst[i] - ts_outburst_model_E[i],
                          s2   = terrs_outburst[i]*terrs_outburst[i];
            
            result += diff*diff/s2 + log(s2);
        }
        result *= -0.5;
        
        constexpr double ln2pi = 1.8378770664093453;
        result -= 0.5*length*ln2pi;
        
        return result;
    }    
}

#endif
