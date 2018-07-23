EXE = gppKer_gpuComplex.ex
SRC = gppKer_gpuComplex.cpp 
#EXE = testSummitReduction.ex
#SRC = testSummitReduction.cpp 
#EXE = testCase2.ex
#SRC = testCase2.cpp 

CXX = xlc++_r
#CXX = CC 
#CXX = g++
#CXX = clang++

LINK = ${CXX}

ifeq ($(CXX),g++)
	CXXFLAGS= -g -O3 -std=c++11 -fopenmp -foffload="-lm" -foffload=nvptx-none
	LINKFLAGS=-fopenmp -foffload="-lm" -foffload=nvptx-none
endif 

ifeq ($(CXX),xlc++_r)
	CXXFLAGS=-std=gnu++11 -g -O3 -qsmp=noauto:omp -qoffload #-qnoipa -qstrict #-Xptxas -v
	LINKFLAGS=-qsmp=noauto:omp -qoffload 
endif 

ifeq ($(CXX),clang++)
	CXXFLAGS=-O3 -std=gnu++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
	LINKFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
endif 

ifeq ($(CXX),icc)
	CXXFLAGS=-O3 -qopenmp -qopt-report=5 -g
	CXXFLAGS+=-xCORE_AVX2
#	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp -dynamic
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)  
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ) $(EXE)

