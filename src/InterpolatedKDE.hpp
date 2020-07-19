#include <vector>
#include <algorithm>
#include <boost/math/interpolators/cubic_b_spline.hpp>
//#include <gsl/gsl_spline.h>
#include <cmath>
#include "Model.hpp"

namespace Opha{

    class InterpolatedKDE{

    private:
        //gsl_interp_accel *acc;
        //gsl_spline *spl;
        
        double mu, sigma;
        double ptmin, ptmax, h;
        
        boost::math::cubic_b_spline<double> spline;
        
    public:
        InterpolatedKDE(const std::vector<double>& pts, const std::vector<double>& vals, const double med, const double std)
            : mu(med), sigma(std), 
              ptmin(*std::min_element(pts.begin(), pts.end())), 
              ptmax(*std::max_element(pts.begin(), pts.end())),
              h((ptmax-ptmin)/(pts.size()-1)),
              spline(vals.begin(), vals.end(), ptmin, h) {
            
            //const unsigned size = pts.size();
            
            //acc = gsl_interp_accel_alloc();
            //spl = gsl_spline_alloc(gsl_interp_cspline, size);
            
            //gsl_spline_init(spl, pts.data(), vals.data(), size);
            
            //ptmin = *std::min_element(pts.begin(), pts.end());
            //ptmax = *std::max_element(pts.begin(), pts.end());
            
            //double h = (ptmax-ptmin)/(size-1);
            
        }
        
        double eval(const double tob){
            if(tob>=ptmin && tob<=ptmax){
                //return gsl_spline_eval(spl, tob, acc);
                return spline(tob);
            }
            else{
                return -0.5*log(2*M_PI*sigma*sigma) - 0.5*((tob-mu)/sigma)*((tob-mu)/sigma);
            }
        }
        
        ~InterpolatedKDE(){
            //gsl_spline_free(spl);
            //gsl_interp_accel_free(acc);
        }
        
    };

    template <typename ModelClass>
    class KDELikelihood{

    private:
        std::vector<double> phis;
        std::vector<InterpolatedKDE> distrs;
        double z;
        const odeint_settings settings;
        
    public:
        KDELikelihood(const double _z)
            : phis(), distrs(), z(_z), settings(odeint_settings::default_settings) {}
        
        void add_distr(const double phi, const std::vector<double>& pts, const std::vector<double>& vals, const double med, const double std){
            phis.push_back(phi);
            const InterpolatedKDE distr {pts, vals, med, std};
            distrs.push_back(std::move(distr));
        }
        
        double operator()(const typename ModelClass::params_t& params);
          
    };

    template <typename ModelClass>
    double KDELikelihood<ModelClass>::operator()(const typename ModelClass::params_t& params){

        const std::vector<double> tobs_model = ModelClass::outburst_times_E(params, phis, z, settings);
        
        const unsigned length = phis.size();
        double result = 0;
        for(unsigned i=0; i<length; i++){
            result += distrs[i].eval(tobs_model[i]);
        }
        
        return result;
    }

}