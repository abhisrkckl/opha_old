#include <array>

namespace Opha{
    
    template <typename ModelClass>
    class PriorTransform{
    
    private:
        constexpr static unsigned nparams = ModelClass::N_PARAMS;
        typedef typename ModelClass::params_t params_t;
        params_t params_min, params_max;
        
    public:
        PriorTransform(params_t _params_min, params_t _params_max) 
            : params_min(_params_min), params_max(_params_max) {}
            
        params_t operator()(params_t cube){
            params_t result;
            for(unsigned i=0; i<nparams; i++){
                if(params_min[i] == params_max[i]){
                    return params_min[i];
                }   
                else{
                    return params_min[i] + (params_max[i]-params_min[i])*cube[i];
                }
            }
            
            return result;
        }
    
    };

}
