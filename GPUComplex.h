#ifndef __GPUCOMPLEX
#define __GPUCOMPLEX

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>


#include "/sw/summitdev/cuda/8.0.54/include/vector_types.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


#define NcoulsKernel 0
#define NgpownKernel 0
#define NcoulsNgpownKernel 0
#define NumberBandsKernel 1

class GPUComplex : public double2{

    private : 
    double re;
    double im;

public:

explicit GPUComplex () {
    re = 0.00;
    im = 0.00;
}


__host__ __device__ explicit GPUComplex(const double& x, const double& y) {
    re = x;
    im = y;
}

GPUComplex(const GPUComplex& src) {
    re = src.re;
    im = src.im;
}

GPUComplex& operator =(const GPUComplex& src) {
    re = src.re;
    im = src.im;

    return *this;
}

GPUComplex& operator +=(const GPUComplex& src) {
    re = src.re + this->re;
    im = src.im + this->im;

    return *this;
}

GPUComplex& operator -=(const GPUComplex& src) {
    re = src.re - this->re;
    im = src.im - this->im;

    return *this;
}

GPUComplex& operator -() {
    re = -this->re;
    im = -this->im;

    return *this;
}

GPUComplex& operator ~() {
    return *this;
}

void print() const {
    printf("( %f, %f) ", this->re, this->im);
    printf("\n");
}

double abs(const GPUComplex& src) {

    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

//    double result = sqrt(re_this+im_this);
    double result = (re_this+im_this);

    return result;

}

double get_real() const
{
    return this->re;
}

double get_imag() const
{
    return this->im;
}

void set_real(double val)
{
    this->re = val;
}

void set_imag(double val) 
{
    this->im = val;
}

    friend inline GPUComplex GPUComplex_square(GPUComplex& src) ;
    friend inline GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    friend inline GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    friend inline double GPUComplex_abs(const GPUComplex& src) ;
    friend inline GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
    friend inline GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    friend inline void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend inline void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend inline GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
    friend inline GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    friend inline double GPUComplex_real( const GPUComplex& src) ;
    friend inline double GPUComplex_imag( const GPUComplex& src) ;

    
//Device Functions 
    friend __device__ const GPUComplex d_GPUComplex_square(GPUComplex& src) ;
    friend __device__ const GPUComplex d_GPUComplex_conj(const GPUComplex& src) ;
    friend __device__ const GPUComplex d_GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    friend __device__ double d_GPUComplex_abs(const GPUComplex& src) ;
    friend __device__ const GPUComplex d_GPUComplex_mult(GPUComplex& a, double b, double c) ;
    friend __device__ const GPUComplex d_GPUComplex_mult(const GPUComplex& a, double b) ;
    friend __device__ void d_GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend __device__ void d_GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend __device__ GPUComplex d_doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
    friend __device__ const GPUComplex d_doublePlusGPUComplex(double a, GPUComplex& src) ;
    friend __device__ double d_GPUComplex_real( const GPUComplex& src) ;
    friend __device__ double d_GPUComplex_imag( const GPUComplex& src) ;
    friend __device__ void d_GPUComplex_plusEquals( GPUComplex& a, const GPUComplex & b); 
    friend __device__ void d_GPUComplex_Equals( GPUComplex& a, const GPUComplex & b); 
    friend __device__ void d_print( const GPUComplex& src) ;
    friend __device__ void ncoulsKernel(GPUComplex& mygpvar1, GPUComplex& wdiff, GPUComplex& aqsntemp_index, GPUComplex& wtilde_array_index, GPUComplex& I_eps_array_index, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc);
    friend __device__ void ncoulsKernel(GPUComplex& mygpvar1, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc);

};
//Inline functions have to be defined in the same file as the declaration

/*
 * Return the square of a complex number 
 */
GPUComplex GPUComplex_square(GPUComplex& src) {
    double re_this = src.re ;
    double im_this = src.im ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
GPUComplex GPUComplex_conj(const GPUComplex& src) {

double re_this = src.re;
double im_this = -1 * src.im;

GPUComplex result(re_this, im_this);
return result;

}


/*
 * Return the product of 2 complex numbers 
 */
GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.re * b.re - a.im*b.im ;
    double im_this = a.re * b.im + a.im*b.re ;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = (re_this+im_this);
//    double result = sqrt(re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.re * b * c, a.im * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
GPUComplex GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.re*b, a.im*b);
   return result;

}

/*
 * Return the complex number a += b * c  
 */
void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re += mult_result.re;
    a.im += mult_result.im;
}

/*
 * Return the complex number a -= b * c  
 */
void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re -= mult_result.re;
    a.im -= mult_result.im;
}


GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    GPUComplex result(a - src.re, 0 - src.im);
    return result;
}

GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.re, 0 + src.im);
    return result;
}

double GPUComplex_real( const GPUComplex& src) {
    return src.re;
}

double GPUComplex_imag( const GPUComplex& src) {
    return src.im;
}

//void gppKernelGPU( GPUComplex* , GPUComplex* , GPUComplex*, GPUComplex* , int , int, int, double* , double* , double* , double* , int , int , int , int, int*, int*);
void gppKernelGPU( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex* aqsmtemp, GPUComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index);

void till_nvbandKernel(GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxtemp, int *inv_igp_index, int *indinv, GPUComplex *wtilde_array, double *wx_array, GPUComplex *I_eps_array, int ncouls, int nvband, int ngpown, int nstart, int nend, double *d_vcoul);

#endif
