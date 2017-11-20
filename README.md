BGW versions introduction
========

The code is divided into CPU and GPU specific versions, i.e., OpenMP3.0 and OpenMP4.5 versions respectively.
Both the versions have implementation specific to std::Complex class and a GPUComplex class.
The GPUComplex class is written specifically to attain performance on GPU versions of the code.
However it also gives good performance on CPU's, better than std::Complex class.
Use intel17 compilers, intel18 does not vectorize the code.

BGW versions 
========
gppKer.cpp - CPU version with std::Complex
gppKer_GPUComplex.cpp - GPU version with GPUComplex
Associated Makefile - Makefile.GPUComplex

gppKer_GPUComplexGCCTarget.cpp - GPU version with GPUComplex class for GCC compiler.
gppKer_GPUComplexXLCTarget.cpp - GPU version with GPUComplex class for XLC compiler.
Associated Makefile - Makefile.GPUComplex

Modify Makefiles as needed.

GPUComplex.h - for GPUComplex class on CPU
GPUComplex_target.cpp - for GPUComplex class on target.

