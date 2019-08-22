#ifndef _Opha_python_hpp_
#define _Opha_python_hpp_

#include <boost/python/numpy.hpp>
#include <boost/python.hpp>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <array>
#include "Likelihood.hpp"

namespace py = boost::python;
namespace np = boost::python::numpy;

std::vector<double> vector_from_pyiter(const py::object& in_iter){

	py::stl_input_iterator<double> start(in_iter), end;
	
	return std::vector<double>(start, end);
}

np::ndarray ndarray_from_vector(const std::vector<double>& in_vec){
	
	const auto double_dtype = np::dtype::get_builtin<double>();
	
	const unsigned length = in_vec.size();
	const auto shape = py::make_tuple(length);
	
	np::ndarray out_arr = np::empty(shape, double_dtype);
	std::copy(in_vec.begin(), in_vec.end(),
		  reinterpret_cast<double*>(out_arr.get_data()));
	  
	return out_arr;
}

template<unsigned arr_size>
std::array<double,arr_size> array_from_pyiter(const py::object& in_iter){

	if(py::len(in_iter) != arr_size){
		throw std::invalid_argument("Input size does not match with required array size.");
	}

	py::stl_input_iterator<double> start(in_iter), end;
	
	std::array<double,arr_size> arr_out;
	
	std::copy(start, end, 
		  arr_out.begin());
	
	return arr_out;
}

template<typename ModelClass>
np::ndarray outburst_times(const py::object& params_iter, const py::object& phis_iter, 
			   const double epsabs, const double epsrel, const double init_step){
	constexpr unsigned N_PARAMS = ModelClass::N_PARAMS;
	
	const typename ModelClass::params_t params{  array_from_pyiter<N_PARAMS>(params_iter)  };
	const std::vector phis = vector_from_pyiter(phis_iter);
	
	const std::vector outburst_ts = ModelClass::outburst_times(params,phis,epsabs,epsrel,init_step);
	
	return ndarray_from_vector(outburst_ts);
	
}

template<typename ModelClass>
np::ndarray outburst_times_x(const py::object& params_samples_iter, const py::object& phis_iter, 
			     const double epsabs, const double epsrel, const double init_step){
	
	constexpr unsigned N_PARAMS = ModelClass::N_PARAMS;
	const unsigned  n_samples = py::len(params_samples_iter),
			n_phis = py::len(phis_iter);
	
	const std::vector phis = vector_from_pyiter(phis_iter);
	
	const auto out_shape = py::make_tuple(n_samples, n_phis);
	const auto double_dtype = np::dtype::get_builtin<double>();
	auto out_ndarray = np::empty(out_shape, double_dtype);
	auto out_ndarray_ptr = reinterpret_cast<double*>(out_ndarray.get_data());
	
	py::stl_input_iterator<py::object> start_params_samples(params_samples_iter), end_params_samples;
	unsigned i=0;
	for(auto params_sample=start_params_samples; params_sample!=end_params_samples; params_sample++, i++){

		const typename ModelClass::params_t params{  array_from_pyiter<N_PARAMS>(*params_sample)  };
		
		const std::vector outburst_ts = ModelClass::outburst_times(params,phis,epsabs,epsrel,init_step);
		
		std::copy(outburst_ts.begin(), outburst_ts.end(), out_ndarray_ptr+(i*n_phis));
	}
		
	return out_ndarray;
}

template<typename ModelClass>
double emission_delay(const py::object& params_iter, const py::object& impact_state_iter){
	constexpr unsigned N_PARAMS        = ModelClass::N_PARAMS,
			   N_STATE_PARAMS  = ModelClass::N_STATE_PARAMS;
	
	const typename ModelClass::params_t params{ array_from_pyiter<N_PARAMS>(params_iter) };
	const typename ModelClass::state_t impact_state{ array_from_pyiter<N_STATE_PARAMS>(impact_state_iter) };
	
	return ModelClass::emission_delay(params, impact_state);
}

template<typename ModelClass>
np::ndarray impacts(const py::object& init_params_iter, const py::object& phis_iter,
		    const double epsabs, const double epsrel, const double init_step){
	
	constexpr unsigned N_PARAMS        = ModelClass::N_PARAMS,
			   N_STATE_PARAMS  = ModelClass::N_STATE_PARAMS;
	
	const typename ModelClass::params_t init_params{ array_from_pyiter<N_PARAMS>(init_params_iter) };
	
	const std::vector phis = vector_from_pyiter(phis_iter);
	
	// std::vector<ModelClass::state_t>
	const auto impacts_vec = ModelClass::impacts(init_params, phis, epsabs, epsrel, init_step);
	const auto impact_vec_ptr = reinterpret_cast<const double*>(impacts_vec.data());
	
	const unsigned  n_phis = py::len(phis_iter);
	const auto out_shape = py::make_tuple(n_phis, N_STATE_PARAMS);
	const auto double_dtype = np::dtype::get_builtin<double>();
	auto out_ndarray = np::empty(out_shape, double_dtype);
	auto out_ndarray_ptr = reinterpret_cast<double*>(out_ndarray.get_data());
	
	std::copy(impact_vec_ptr, impact_vec_ptr+n_phis*N_STATE_PARAMS,
		  out_ndarray_ptr);
	
	/*
	unsigned i=0;
	for(const auto& impact : impacts_vec){
		std::copy(impact.begin(), impact.end(), 
			  out_ndarray_ptr+(i*N_STATE_PARAMS) );
		i++;
	}*/
	
	return out_ndarray;
	
}

template<typename ModelClass>
np::ndarray coord_and_velocity(const py::object& params_iter, const py::object& state_iter, const double phi){

	constexpr unsigned N_PARAMS        = ModelClass::N_PARAMS,
			   N_STATE_PARAMS  = ModelClass::N_STATE_PARAMS;

	const typename ModelClass::params_t params{ array_from_pyiter<N_PARAMS>(params_iter) };
	const typename ModelClass::state_t state = array_from_pyiter<N_STATE_PARAMS>(state_iter);
	
	const std::array coord_and_velocity_res = ModelClass::coord_and_velocity(params, state, phi);
	
	const auto out_shape = py::make_tuple(N_STATE_PARAMS);
	const auto double_dtype = np::dtype::get_builtin<double>();
	auto out_ndarray = np::empty(out_shape, double_dtype);
	auto out_ndarray_ptr = reinterpret_cast<double*>(out_ndarray.get_data());
	
	std::copy(coord_and_velocity_res.begin(), coord_and_velocity_res.end(), 
		  out_ndarray_ptr);
		
	return out_ndarray;
		
}

template<typename ModelClass>
struct Likelihood_wrap {

private:
	Opha::Likelihood<ModelClass> likelihood;

public:
	Likelihood_wrap(const py::object& _phis, const py::object& _ts_outburst, const py::object& _terrs_outburst, const double _z)
			: likelihood(vector_from_pyiter(_phis), 
			  	     vector_from_pyiter(_ts_outburst), 
			  	     vector_from_pyiter(_terrs_outburst),
			  	     _z) {
		
		if( !(py::len(_phis)==py::len(_ts_outburst) && py::len(_phis)==py::len(_terrs_outburst)) ){
			throw std::invalid_argument("The array lengths do not match.");
		}
	}
	
	Likelihood_wrap(const py::object& _phis, const py::object& _ts_outburst, const py::object& _terrs_outburst, 
		        const double _z,
		        const double _epsabs, const double _epsrel, const double _init_step)
			: likelihood(vector_from_pyiter(_phis), 
			  	     vector_from_pyiter(_ts_outburst), 
			  	     vector_from_pyiter(_terrs_outburst),
			  	     _z,
			  	     _epsabs, _epsrel, _init_step) {
		
		if( !(py::len(_phis)==py::len(_ts_outburst) && py::len(_phis)==py::len(_terrs_outburst)) ){
			throw std::invalid_argument("The array lengths do not match.");
		}
	}
	
	double operator()(const py::object& params_iter) const{
		constexpr unsigned N_PARAMS = ModelClass::N_PARAMS;
		const typename ModelClass::params_t params{  array_from_pyiter<N_PARAMS>(params_iter)  };
		return likelihood(params); 
	}
};

#define NEW_MODEL(ModelClass, model_str)						\
											\
	BOOST_PYTHON_MODULE(ModelClass##_py)						\
	{										\
		Py_Initialize();							\
		np::initialize();							\
		py::def("outburst_times", outburst_times<ModelClass>);			\
		py::def("outburst_times_x", outburst_times_x<ModelClass>);		\
		py::def("description", ModelClass::description);				\
		py::def("emission_delay",emission_delay<ModelClass>);			\
		py::def("impacts", impacts<ModelClass>);				\
		py::def("coord_and_velocity", coord_and_velocity<ModelClass>);				\
		py::scope().attr("N_PARAMS") 	    = (int)ModelClass::N_PARAMS;	\
		py::scope().attr("N_STATE_PARAMS")  = (int)ModelClass::N_STATE_PARAMS;	\
		py::scope().attr("N_CONST_PARAMS")  = (int)ModelClass::N_CONST_PARAMS;	\
		py::scope().attr("N_BINARY_PARAMS") = (int)ModelClass::N_BINARY_PARAMS;	\
		py::scope().attr("N_DELAY_PARAMS")  = (int)ModelClass::N_DELAY_PARAMS;	\
		py::class_<Likelihood_wrap<ModelClass> >("Likelihood", py::init<np::ndarray, np::ndarray, np::ndarray, double>()) \
			.def(py::init<np::ndarray, np::ndarray, np::ndarray, double, double, double, double>())	\
			.def("__call__", &Likelihood_wrap<ModelClass>::operator());	\
	}

#endif
