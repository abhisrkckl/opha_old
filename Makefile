CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python2.7/
INCLDIR = include/
INCLUDES = -I$(INCLDIR) $(PYTHONI)

LIBS = -lboost_python -lboost_numpy

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Model.hpp $(INCLDIR)/Likelihood.hpp $(INCLDIR)/python.hpp

.PHONY: models
all: py/NoSpin_py.so py/Spin_py.so 

py/NoSpin_py.so: Models/NoSpin.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Spin_py.so: Models/Spin.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm py/*_py.so
