EXE = gppKer.ex
SRC = gppKer.cpp 
#EXE = gppKer_gpuComplex.ex
#SRC = gppKer_gpuComplex.cpp 

#CXX = xlc++
#CXX = g++
CXX = CC

LINK = ${CXX}

ifeq ($(CXX),CC)
	CXXFLAGS= -g -O3 -qopenmp -qopt-report=5 -std=c++11
	#CXXFLAGS+=-xCORE_AVX2
	CXXFLAGS+=-xMIC-AVX512
	LINKFLAGS=-qopenmp -dynamic
endif 

ifeq ($(CXX),g++)
	CXXFLAGS= -g -O3 -std=c++11 -fopenmp -foffload="-lm" -foffload=nvptx-none
	LINKFLAGS=-fopenmp
endif 

ifeq ($(CXX),xlc++)
	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp=noauto:omp -qoffload #-Xptxas -v
	LINKFLAGS=-qsmp=noauto:omp -qoffload 
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

