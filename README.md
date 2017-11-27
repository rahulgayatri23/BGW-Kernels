BGW versions introduction
========

Berkeley GW kernel with the following implementations

Fortran - OpenMP3.0
C++ - OpenMP{3.0, 4.5}
C++ - Cuda

Each implementation has two instances : 
    1. With std::complex
    2. With an Optimized hand-coded Complex class 

The optimized Complex class for GPU is written using the Cuda Complex class which is a data structure comprised of double2 vector type. 
Use intel17 compilers, intel18 does not vectorize the code.

BGW versions 
========
gppKer.cpp - CPU version with std::Complex
gppKer_GPUComplex.cpp - CPU version with Optimized complex class 
Associated Makefile - Makefile

gppKer_GPUComplexGCCTarget.cpp - GPU version with GPUComplex class for GCC compiler.
gppKer_GPUComplexXLCTarget.cpp - GPU version with GPUComplex class for XLC compiler.
Associated Makefile - Makefile.GPUComplex


gppKer_cuComplex.cpp - GPU version with Optimized Cuda complex class
cuComplex.cu - Cuda Kernels

Modify Makefiles as needed.
