CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python3.6m/
INCLDIR = src/
MODELDIR = src/Models
INCLUDES = -I$(INCLDIR) $(PYTHONI)
PYDIR = scripts/

LIBS = -lboost_python3 -lboost_numpy3

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Model.hpp $(INCLDIR)/Likelihood.hpp $(INCLDIR)/python.hpp

.PHONY: all
all: $(PYDIR)/Model6_py.so $(PYDIR)/Model8_py.so  

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
