#ifndef __CudaComplex
#define __CudaComplex

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>


#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#define NumberBandsKernel 1
#define NgpownKernel 0

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

inline const cuDoubleComplex cuDoubleComplex_square(cuDoubleComplex& src) {
    return make_cuDoubleComplex(src.x*src.x - src.y*src.y, 2*src.x*src.y);
}

/*
Return the conjugate of a complex number 
*/
inline cuDoubleComplex cuDoubleComplex_conj(cuDoubleComplex& src) {
return make_cuDoubleComplex(src.x, -src.y);
}


/*
 * Return the product of 2 complex numbers 
 */
inline const cuDoubleComplex cuDoubleComplex_product(const cuDoubleComplex& a, const cuDoubleComplex& b) {
    return make_cuDoubleComplex(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}


/*
 * Return the absolute of a complex number 
 */
inline double cuDoubleComplex_abs(const cuDoubleComplex& src) {
    return sqrt(src.x * src.x + src.y * src.y);
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
const inline cuDoubleComplex cuDoubleComplex_mult(cuDoubleComplex& a, double b, double c) {
    return make_cuDoubleComplex(a.x * b * c, a.y * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
const inline cuDoubleComplex cuDoubleComplex_mult(const cuDoubleComplex& a, double b) {
   return make_cuDoubleComplex(a.x*b, a.y*b);

}

/*
 * Return the complex number a += b * c  
 */
inline void cuDoubleComplex_fma(cuDoubleComplex& a, const cuDoubleComplex& b, const cuDoubleComplex& c) {
    a.x += b.x * c.x - b.y*c.y ;
    a.y += b.x * c.y + b.y*c.x ;
}

inline cuDoubleComplex doubleMinuscuDoubleComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a-src.x, -src.y);
}

inline cuDoubleComplex doubleMinuscuComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a - src.x, -src.y);
}

inline cuDoubleComplex doublePluscuComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a + src.x, src.y);
}


inline double cuDoubleComplex_real( const cuDoubleComplex& src) {
    return src.x;
}

inline double cuDoubleComplex_imag( const cuDoubleComplex& src) {
    return src.y;
}

inline void cuDoubleComplex_plusEquals( cuDoubleComplex& a, const cuDoubleComplex & b) {
    a.y += b.x;
    a.x += b.y;
}

inline void cuDoubleComplex_Equals( cuDoubleComplex& a, const cuDoubleComplex & b) {
    a.x = b.x;
    a.y = b.y;
}

void gppKernelGPU( cuDoubleComplex *wtilde_array, cuDoubleComplex *aqsntemp, cuDoubleComplex* aqsmtemp, cuDoubleComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index);

void till_nvbandKernel(cuDoubleComplex *aqsmtemp, cuDoubleComplex* aqsntemp, cuDoubleComplex* asxtemp, int *inv_igp_index, int *indinv, cuDoubleComplex *wtilde_array, double *wx_array, cuDoubleComplex *I_eps_array, int ncouls, int number_bands, int ngpown, int nstart, int nend, double* vcoul);
#endif
