CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

PYTHONI = -I/usr/include/python2.7/
INCLDIR = src/
MODELDIR = src/Models
INCLUDES = -I$(INCLDIR) $(PYTHONI)

LIBS = -lboost_python -lboost_numpy

HEADERS = $(INCLDIR)/ipow.hpp $(INCLDIR)/Opha.hpp $(INCLDIR)/Model.hpp $(INCLDIR)/Likelihood.hpp $(INCLDIR)/python.hpp

.PHONY: models
#all: py/BinX_PN_py.so py/Newtonian_py.so py/Model3_py.so py/Model4_py.so py/Model5_py.so py/Model6_py.so py/Model7_py.so
all: py/Model6_py.so py/Model7_py.so py/Model8_py.so  py/Model9_py.so

#py/BinX_PN_py.so: Models/BinX_PN.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Model3_py.so: Models/Model3.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Model4_py.so: Models/Model4.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Model5_py.so: Models/Model5.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model6_py.so: $(MODELDIR)/Model6.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Model61_py.so: Models/Model61.cpp Models/PN.cpp  $(HEADERS)
#	$(CXX) -shared $< Models/PN.cpp -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model7_py.so: $(MODELDIR)/Model7.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model8_py.so: $(MODELDIR)/Model8.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

py/Model9_py.so: $(MODELDIR)/Model9.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

#py/Newtonian_py.so: Models/Newtonian.cpp $(HEADERS)
#	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm py/BinX_PN_py.so py/Newtonian_py.so py/Model*_py.so
