CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python2.7/
INCLDIR = src/
INCLUDES = -I$(INCLDIR) $(PYTHONI)
MODELDIR = src/Models
PYDIR = scripts/

LIBS = -lboost_python -lboost_numpy

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Model.hpp $(INCLDIR)/Likelihood.hpp $(INCLDIR)/python.hpp

.PHONY: models
all: $(PYDIR)/NoSpin_py.so $(PYDIR)/Spin_py.so 

$(PYDIR)/NoSpin_py.so: $(MODELDIR)/NoSpin.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

$(PYDIR)/Spin_py.so: $(MODELDIR)/Spin.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm $(PYDIR)/*_py.so
