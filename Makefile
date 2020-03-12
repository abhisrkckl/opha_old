CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python2.7/
INCLDIR = src/
MODELDIR = src/Models
INCLUDES = -I$(INCLDIR) $(PYTHONI)
PYDIR = scripts/

LIBS = -lboost_python -lboost_numpy

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Model.hpp $(INCLDIR)/Likelihood.hpp $(INCLDIR)/python.hpp

.PHONY: all
all: $(PYDIR)/Model6_py.so $(PYDIR)/Model7_py.so $(PYDIR)/Model8_py.so  $(PYDIR)/Model9_py.so

$(PYDIR)/Model6_py.so: $(MODELDIR)/Model6.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#$(PYDIR)/Model7_py.so: $(MODELDIR)/Model7.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

$(PYDIR)/Model8_py.so: $(MODELDIR)/Model8.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#$(PYDIR)/Model9_py.so: $(MODELDIR)/Model9.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm $(PYDIR)/Model*_py.so
