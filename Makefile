CXX = g++ -std=c++17 
CXXFLAGS = -fPIC -Wall -Wno-unused-variable -O2

INCLUDES = -Isrc/ -I/home/abhimanyu/anaconda3/envs/opha/include/ -I/home/abhimanyu/anaconda3/envs/opha/include/python3.6m/
MODELDIR = src/Models
PYDIR = scripts/

LIBS = -L/home/abhimanyu/anaconda3/envs/opha/lib -lboost_python36 -lboost_numpy36

HEADERS = src/ipow.hpp src/Opha.hpp src/Model.hpp src/Likelihood.hpp src/InterpolatedKDE.hpp  src/python.hpp 

.PHONY: models
all: $(PYDIR)/NoSpin_py.so $(PYDIR)/Spin_py.so 

$(PYDIR)/NoSpin_py.so: $(MODELDIR)/NoSpin.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

$(PYDIR)/Spin_py.so: $(MODELDIR)/Spin.cpp $(HEADERS)
	$(CXX) -shared $< -o $@ $(CXXFLAGS) $(LIBS) $(INCLUDES)

.PHONY: clean
clean:
	rm $(PYDIR)/*_py.so
