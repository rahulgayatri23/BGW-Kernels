//#include <iostream>
//#include <cstdlib>
//
//#include <iomanip>
//#include <cmath>
//#include <complex>
//#include <omp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
using namespace std;


#define CUDASPACE 1
#define OPENMPSPACE 0
#define CUDAUVM 0
#define SERIAL 0
#define THREADS 0

#if OPENMPSPACE
#include "Complex.h"
        typedef Kokkos::OpenMP   ExecSpace;
        typedef Kokkos::OpenMP        MemSpace;
        typedef Kokkos::LayoutRight  Layout;
#endif

#if CUDASPACE
#include "GPUComplex.h"
        typedef Kokkos::Cuda     ExecSpace;
        typedef Kokkos::CudaSpace     MemSpace;
        typedef Kokkos::LayoutLeft   Layout;
#endif

#if CUDAUVM
#include "GPUComplex.h"
        typedef Kokkos::Cuda     ExecSpace;
        typedef Kokkos::CudaUVMSpace  MemSpace;
        typedef Kokkos::LayoutLeft   Layout;
#endif

#if SERIAL
#include "Complex.h"
        typedef Kokkos::Serial   ExecSpace;
        typedef Kokkos::HostSpace     MemSpace;
#endif

#if THREADS
#include "Complex.h"
        typedef Kokkos::Threads  ExecSpace;
        typedef Kokkos::HostSpace     MemSpace;
#endif

typedef Kokkos::RangePolicy<ExecSpace>  range_policy;
typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;

typedef Kokkos::View<GPUComplex, Layout, MemSpace>   ViewScalarTypeComplex;
typedef Kokkos::View<GPUComplex*, Layout, MemSpace>   ViewVectorTypeComplex;
typedef Kokkos::View<GPUComplex**, Layout, MemSpace>  ViewMatrixTypeComplex;

typedef Kokkos::View<int*, Layout, MemSpace>   ViewVectorTypeInt;
typedef Kokkos::View<double*, Layout, MemSpace>   ViewVectorTypeDouble;

typedef Kokkos::View<int, Layout, MemSpace>   ViewScalarTypeInt;
typedef Kokkos::View<double, Layout, MemSpace>   ViewScalarTypeDouble;

//KOKKOS_INLINE_FUNCTION
struct achtempStruct 
{
    double achtemp_re[3];
    double achtemp_im[3];
KOKKOS_INLINE_FUNCTION
    void operator+=(achtempStruct const& other) 
    {
        for (int i = 0; i < 3; ++i) 
        {
            achtemp_re[i] += other.achtemp_re[i];
            achtemp_im[i] += other.achtemp_im[i];
        }
    }
KOKKOS_INLINE_FUNCTION
    void operator+=(achtempStruct const volatile& other) volatile 
    {
        for (int i = 0; i < 3; ++i) 
        {
            achtemp_re[i] += other.achtemp_re[i];
            achtemp_im[i] += other.achtemp_im[i];
        }
    }
};
