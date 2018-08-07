/*
Templated CustomComplex class that represents a complex class comprised of  any type of real and imaginary types.
*/
#ifndef __CustomComplex
#define __CustomComplex

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <sys/time.h>

#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define nstart 0
#define nend 3

#define singleDim 1

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
    // Comment away if needed. - Rahul - commented the below deviceSynchronize
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}


class CustomComplex : public double2{

    private : 
    //The x and y are now available from the double2 data type
//    re x;
//    im y;

    public:
    explicit CustomComplex () {
        x = 0.00;
        y = 0.00;
    }


__host__ __device__ explicit CustomComplex(const double& a, const double& b) {
        x = a;
        y = b;
    }

__host__ __device__ CustomComplex(const CustomComplex& src) {
        x = src.x;
        y = src.y;
    }

__host__ __device__ CustomComplex& operator =(const CustomComplex& src) {
        x = src.x;
        y = src.y;

        return *this;
    }

__host__ __device__ CustomComplex& operator +=(const CustomComplex& src) {
        x = src.x + this->x;
        y = src.y + this->y;

        return *this;
    }

__host__ __device__ CustomComplex& operator -=(const CustomComplex& src) {
       x = src.x - this->x;
       y = src.y - this->y;

       return *this;
   }

__host__ __device__ CustomComplex& operator -() {
        x = -this->x;
        y = -this->y;

        return *this;
    }

__host__ __device__ CustomComplex& operator ~() {
        return *this;
    }

    void print() const {
        printf("( %f, %f) ", this->x, this->y);
        printf("\n");
    }

    double get_real() const
    {
        return this->x;
    }

    double get_imag() const
    {
        return this->y;
    }

    void set_real(double val)
    {
        this->x = val;
    }

    void set_imag(double val) 
    {
        this->y = val;
    }

// 6 flops
__host__ __device__ friend inline CustomComplex operator *(const CustomComplex &a, const CustomComplex &b) {
        double x_this = a.x * b.x - a.y*b.y ;
        double y_this = a.x * b.y + a.y*b.x ;
        CustomComplex result(x_this, y_this);
        return (result);
    }

//2 flops
    __host__ __device__ friend inline CustomComplex operator *(const CustomComplex &a, const double &b) {
       CustomComplex result(a.x*b, a.y*b);
       return result;
    }

//2 flops
    __host__ __device__ friend inline CustomComplex operator -(const double &a, CustomComplex& src) {
        CustomComplex result(a - src.x, 0 - src.y);
        return result;
    }

    __host__ __device__ friend inline CustomComplex operator +(const double &a, CustomComplex& src) {
        CustomComplex result(a + src.x, src.y);
        return result;
    }

    __host__ __device__ friend inline CustomComplex CustomComplex_conj(const CustomComplex& src) ;

    __host__ __device__ friend inline double CustomComplex_abs(const CustomComplex& src) ;

    __host__ __device__ friend inline double CustomComplex_real( const CustomComplex& src) ;

    __host__ __device__ friend inline double CustomComplex_imag( const CustomComplex& src) ;
};

/*
 * Return the conjugate of a complex number 
 1flop
 */
inline CustomComplex CustomComplex_conj(const CustomComplex& src) {

    double re_this = src.x;
    double im_this = -1 * src.y;

    CustomComplex result(re_this, im_this);
    return result;

}

/*
 * Return the absolute of a complex number 
 */
inline double CustomComplex_abs(const CustomComplex& src) {
    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

    double result = sqrt(re_this+im_this);
    return result;
}

/*
 * Return the real part of a complex number 
 */
inline double CustomComplex_real( const CustomComplex& src) {
    return src.x;
}

/*
 * Return the imaginary part of a complex number 
 */
inline double CustomComplex_imag( const CustomComplex& src) {
    return src.y;
}

//Cuda kernel declarations
void d_noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, int stride);
#endif
