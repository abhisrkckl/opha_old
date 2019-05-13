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
np::ndarray outburst_times(const py::object& params_iter, const py::object& phis_iter){
	constexpr unsigned N_PARAMS = ModelClass::N_PARAMS;
	
	const typename ModelClass::params_t params{  array_from_pyiter<N_PARAMS>(params_iter)  };
	const std::vector phis = vector_from_pyiter(phis_iter);
	
	const std::vector outburst_ts = ModelClass::outburst_times(params,phis);
	
	return ndarray_from_vector(outburst_ts);
	
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
		py::scope().attr("N_PARAMS") 	    = (int)ModelClass::N_PARAMS;	\
		py::scope().attr("N_STATE_PARAMS")  = (int)ModelClass::N_STATE_PARAMS;	\
		py::scope().attr("N_CONST_PARAMS")  = (int)ModelClass::N_CONST_PARAMS;	\
		py::scope().attr("N_BINARY_PARAMS") = (int)ModelClass::N_BINARY_PARAMS;	\
		py::scope().attr("N_DELAY_PARAMS")  = (int)ModelClass::N_DELAY_PARAMS;	\
		py::class_<Likelihood_wrap<ModelClass> >("Likelihood", py::init<np::ndarray, np::ndarray, np::ndarray, double>()) \
			.def("__call__", &Likelihood_wrap<ModelClass>::operator());	\
	}

#endif
