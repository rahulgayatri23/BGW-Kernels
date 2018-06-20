#EXE = gppKer.ex
#SRC = gppKer.cpp 
EXE = gppKer_gpuComplex.ex
SRC = gppKer_gpuComplex.cpp 

#EXE = test2.ex
#SRC = test2.cpp 

CXX = xlc++_r
#CXX = g++
#CXX = CC

LINK = ${CXX}

ifeq ($(CXX),CC)
	CXXFLAGS= -g -O3 -qopenmp -qopt-report=5 -std=c++11
#    CXXFLAGS+=-I /usr/common/software/likwid/4.3.0/include/ -DLIKWID_PERFMON
    CXXFLAGS+=-I /usr/common/software/likwid/4.3.0/include/ -DUSE_VTUNE -I${VTUNE_AMPLIFIER_XE_2018_DIR}/include -DLIKWID_PERFMON
	CXXFLAGS+=-xCORE-AVX2
	#CXXFLAGS+=-xMIC-AVX512
	LINKFLAGS=-qopenmp -dynamic
    LINKFLAGS+=-L /usr/common/software/likwid/4.3.0/lib -llikwid
endif 

ifeq ($(CXX),g++)
	CXXFLAGS= -g -O3 -std=c++11 -fopenmp 
	LINKFLAGS=-fopenmp
endif 

ifeq ($(CXX),xlc++_r)
	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp
	LINKFLAGS=-qsmp
endif 

ifeq ($(CXX),clang++)
	CXXFLAGS=-O3 -std=gnu++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
	LINKFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)  
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ) $(EXE)

