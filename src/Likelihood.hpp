#ifndef _Opha_Likelihood_hpp_
#define _Opha_Likelihood_hpp_

#include <vector>
#include <cmath>
#include "Model.hpp"

typedef std::vector<double> DoubleVector;

namespace Opha {

    template <typename ModelClass>
    struct Likelihood {
        
        const DoubleVector phis, ts_outburst, terrs_outburst;
        const double z;
        const odeint_settings settings;
        
        Likelihood(const DoubleVector& _phis, const DoubleVector& _ts_outburst, const DoubleVector& _terrs_outburst, const double _z)
            : phis(_phis), ts_outburst(_ts_outburst), terrs_outburst(_terrs_outburst), 
              z(_z), settings(odeint_settings::default_settings) {  }
        
        Likelihood(const DoubleVector& _phis, const DoubleVector& _ts_outburst, const DoubleVector& _terrs_outburst, const double _z, const odeint_settings& _settings)
            : phis(_phis), ts_outburst(_ts_outburst), terrs_outburst(_terrs_outburst), 
              z(_z), settings(_settings)  { }
        
        double operator()(const typename ModelClass::params_t& params) const;
    };

    /*template <typename ModelClass>
    double Likelihood<ModelClass>::operator()(const typename ModelClass::params_t& params) const{

        const DoubleVector ts_outburst_model = ModelClass::outburst_times(params, phis, settings);
        
        constexpr unsigned IDX_t0 = ModelClass::N_STATE_PARAMS-1;
        const double &t0 = params.state()[IDX_t0];
        
        const unsigned length = phis.size();
        double result = 0;
        for(unsigned i=0; i<length; i++){
            
            const double     ts_outburst_model_redshifted_i = t0 + (ts_outburst_model[i]-t0)*(1+z);
            
            const double     diff = ts_outburst[i] - ts_outburst_model_redshifted_i,        // - ts_outburst_model[i],
                            s2   = terrs_outburst[i]*terrs_outburst[i];
            
            result += diff*diff/s2 + log(s2);
        }
        result *= -0.5;
        
        constexpr double ln2pi = 1.8378770664093453;
        result -= 0.5*length*ln2pi;
        
        return result;
    } */
    
    template <typename ModelClass>
    double Likelihood<ModelClass>::operator()(const typename ModelClass::params_t& params) const{

        const DoubleVector ts_outburst_model = ModelClass::outburst_times_E(params, phis, z, settings);
        
        const unsigned length = phis.size();
        double result = 0;
        for(unsigned i=0; i<length; i++){
            
            const double     diff = ts_outburst[i] - ts_outburst_model[i],
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
