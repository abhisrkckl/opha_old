CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python2.7/
INCLDIR = include/
INCLUDES = -I$(INCLDIR) $(PYTHONI)

LIBS = -lboost_python -lboost_numpy

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Opha/Model.hpp $(INCLDIR)/Opha/Likelihood.hpp $(INCLDIR)/Opha/python.hpp

.PHONY: models
all: py/BinX_PN_py.so py/Newtonian_py.so

py/BinX_PN_py.so: Models/BinX_PN.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Newtonian_py.so: Models/Newtonian.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm py/BinX_PN_py.so py/Newtonian_py.so
