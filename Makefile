KOKKOS_PATH = ${HOME}/Kokkos/kokkos
#KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "gppKerKokkosOpenMP"

SRC = gppKerKokkosOpenMP.cpp
CXX=CC

default: build
	echo "Start Build"

ifeq ($(CXX),CC)
	CXXFLAGS=-O3 #-std=c++11
	#CXXFLAGS+=-xCORE_AVX2
	CXXFLAGS+=-xMIC_AVX512
#	LINKFLAGS=-qopenmp -dynamic
    EXE = ${EXE_NAME}.host
    KOKKOS_ARCH = "knl"
endif 

#ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
#CXX = ${KOKKOS_PATH}/config/nvcc_wrapper
#EXE = ${EXE_NAME}.cuda
#KOKKOS_ARCH = "SNB,Kepler30"
#KOKKOS_CUDA_OPTIONS = "enable_lambda"
#else
#CXX =CC
#EXE = ${EXE_NAME}.host
#KOKKOS_ARCH = "knl"
#endif

#CXXFLAGS = -O3 -xCORE_AVX2 -qopt-report=5 
LINK = ${CXX}
LINKFLAGS = 

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
