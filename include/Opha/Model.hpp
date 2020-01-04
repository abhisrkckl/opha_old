#ifndef _Opha_Model_hpp_
#define _Opha_Model_hpp_

#include <boost/numeric/odeint.hpp>
#include <array>
#include <vector>
#include <string>
#include <iostream>

namespace Opha {

    template <unsigned N_STATE, unsigned N_CONSTS, unsigned N_BIN, unsigned N_DELAY, unsigned DET=0>
    struct Model {
    
        static std::string description();
        
        enum {  N_STATE_PARAMS  = N_STATE, 
                N_CONST_PARAMS  = N_CONSTS, 
                N_BINARY_PARAMS = N_BIN, 
                N_DELAY_PARAMS  = N_DELAY, 
                N_PARAMS        = N_STATE + N_CONSTS + N_BIN + N_DELAY
        };
        
        static_assert(N_STATE_PARAMS>0 && N_PARAMS>=4, "N_STATE must be non-zero and N_PARAMS must be >=4.");
        
        typedef std::array<double,N_STATE_PARAMS>   state_t;        // The last element of state_t must be time.
        typedef std::array<double,N_CONST_PARAMS>   const_params_t;
        typedef std::array<double,N_BINARY_PARAMS>  bin_params_t;
        typedef std::array<double,N_DELAY_PARAMS>   delay_params_t;
        
        class params_t;
        
        struct ODE_system{
            const params_t params;
            
            ODE_system(const params_t& _params) : params(_params) {}
            
            // To be implemented separately for each model
            void operator()(const state_t& state, state_t& derivatives_out, const double /*phi*/ ) const;
        };
        
        static std::vector<state_t> impacts(const params_t& params, const std::vector<double>& phis,
                                            const double epsabs, const double epsrel, const double init_step);
        
        static double emission_delay(const params_t& params, const state_t& impact_state, const double phi);
        
        static std::vector<double> outburst_times(const params_t& params, const std::vector<double>& phis,
                                                  const double epsabs, const double epsrel, const double init_step){
            
            const std::vector<state_t> impacts_ = impacts(params, phis, epsabs, epsrel, init_step);
            
            const unsigned length = phis.size();
            std::vector<double> result(length);
            
            for(unsigned i=0; i<length; i++){
                const state_t& impact_state = impacts_[i];
                
                constexpr unsigned IDX_TIME = N_STATE_PARAMS-1;
                
                result[i] = impact_state[IDX_TIME] + emission_delay(params, impact_state, phis[i]);
            }
            
            return result;        
        }
        
        static std::array<double,3> coord_and_velocity(const params_t& params, const state_t& state, const double phi);
        
    };

    template <unsigned N_STATE, unsigned N_COM, unsigned N_BIN, unsigned N_DELAY, unsigned DET>
    class Model<N_STATE,N_COM,N_BIN,N_DELAY,DET>::params_t{

    private:
        // [state | consts | binary | delay ]
        std::array<double,N_PARAMS> params;
        
        template<typename slice_t, unsigned IDX_SLICE>
        const slice_t& slice_params() const{
        
            constexpr unsigned slice_size = std::tuple_size<slice_t>::value;
            
            static_assert(IDX_SLICE+slice_size <= N_PARAMS, "slice out of bounds.");
            
            if constexpr(slice_size>0){
                return *reinterpret_cast<const slice_t*>(&params[IDX_SLICE]);
            }
            else{
                static constexpr slice_t empty_slice;
                return empty_slice;
            }
        }
        
        enum {  IDX_STATE   = 0,
                IDX_COM     = N_STATE, 
                IDX_BIN     = N_STATE + N_COM, 
                IDX_DELAY   = N_STATE + N_COM + N_BIN 
        };

    public:
        params_t(const std::array<double,N_PARAMS> &_params) : params(_params) {}
        
        const state_t& state() const{
            return slice_params<state_t,IDX_STATE>();
        }
        const const_params_t& const_params() const{
            return slice_params<const_params_t,IDX_COM>();
        }
        const bin_params_t& bin_params() const{
            return slice_params<bin_params_t,IDX_BIN>();
        }
        const delay_params_t& delay_params() const{
            return slice_params<delay_params_t,IDX_DELAY>();
        }
    };

    template <unsigned N_STATE, unsigned N_COM, unsigned N_BIN, unsigned N_DELAY, unsigned DET>
    auto Model<N_STATE,N_COM,N_BIN,N_DELAY,DET>::impacts(const params_t &init_params, const std::vector<double> &phis,
                                                         const double epsabs, const double epsrel, const double init_step) 
    -> std::vector<state_t> {

        namespace boost_ode = boost::numeric::odeint;
        
        constexpr double phi0  = 0;
        const state_t &Y0  = init_params.state();
        
        const unsigned length = phis.size();
        std::vector<state_t> result(length);
        
        state_t Y = Y0;    // [x0, e0, u0, t0]
        double phi_init = phi0;
        
        typedef boost_ode::runge_kutta_fehlberg78<state_t> RKF78_error_stepper_t;
        const auto control = boost_ode::make_controlled<RKF78_error_stepper_t>(epsabs, epsrel);
        const ODE_system system{init_params};
        
        for(unsigned i=0; i<length; i++){
            boost_ode::integrate_adaptive(control, system, Y, phi_init, phis[i], init_step);
            result[i] = Y;
            
            phi_init = phis[i];
        }
        
        return result;
    }
}

#endif
