CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python2.7/
INCLDIR = include/
INCLUDES = -I$(INCLDIR) $(PYTHONI)

LIBS = -lboost_python -lboost_numpy

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Opha/Model.hpp $(INCLDIR)/Opha/Likelihood.hpp $(INCLDIR)/Opha/python.hpp

.PHONY: models
#all: py/BinX_PN_py.so py/Newtonian_py.so py/Model3_py.so py/Model4_py.so py/Model5_py.so py/Model6_py.so py/Model7_py.so
all: py/Model4_py.so py/Model6_py.so py/Model7_py.so

#py/BinX_PN_py.so: Models/BinX_PN.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Model3_py.so: Models/Model3.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model4_py.so: Models/Model4.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Model5_py.so: Models/Model5.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model6_py.so: Models/Model6.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model7_py.so: Models/Model7.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Newtonian_py.so: Models/Newtonian.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm py/BinX_PN_py.so py/Newtonian_py.so py/Model*_py.so
