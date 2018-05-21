#ifndef __GPUCOMPLEX
#define __GPUCOMPLEX

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <ctime>
#include <chrono>


#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

class GPUComplex : public double2{

    private : 

public:
    double2 complNum;

explicit GPUComplex () {
    complNum.x = 0.00;
    complNum.y = 0.00;
}


__host__ __device__ explicit GPUComplex(const double& x, const double& y) {
    complNum.x = x;
    complNum.y = y;
}

GPUComplex(const GPUComplex& src) {
    complNum.x = src.complNum.x;
    complNum.y = src.complNum.y;
}

GPUComplex& operator =(const GPUComplex& src) {
    complNum.x = src.complNum.x;
    complNum.y = src.complNum.y;

    return *this;
}

GPUComplex& operator +=(const GPUComplex& src) {
    complNum.x = src.complNum.x + this->complNum.x;
    complNum.y = src.complNum.y + this->complNum.y;

    return *this;
}

GPUComplex& operator -=(const GPUComplex& src) {
    complNum.x = src.complNum.x - this->complNum.x;
    complNum.y = src.complNum.y - this->complNum.y;

    return *this;
}

GPUComplex& operator -() {
    complNum.x = -this->complNum.x;
    complNum.y = -this->complNum.y;

    return *this;
}

GPUComplex& operator ~() {
    return *this;
}

void print() const {
    printf("( %f, %f) ", this->complNum.x, this->complNum.y);
    printf("\n");
}

double abs(const GPUComplex& src) {

    double re_this = src.complNum.x * src.complNum.x;
    double im_this = src.complNum.y * src.complNum.y;

    double result = (re_this+im_this);
    return result;
}

double get_real() const
{
    return this->complNum.x;
}

double get_imag() const
{
    return this->complNum.y;
}

void set_real(double val)
{
    this->complNum.x = val;
}

void set_imag(double val) 
{
    this->complNum.y = val;
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
    double re_this = src.complNum.x ;
    double im_this = src.complNum.y ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
GPUComplex GPUComplex_conj(const GPUComplex& src) {

    double re_this = src.complNum.x;
    double im_this = -1 * src.complNum.y;

    GPUComplex result(re_this, im_this);
    return result;
}


/*
 * Return the product of 2 complex numbers 
 */
GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.complNum.x * b.complNum.x - a.complNum.y*b.complNum.y ;
    double im_this = a.complNum.x * b.complNum.y + a.complNum.y*b.complNum.x ;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.complNum.x * src.complNum.x;
    double im_this = src.complNum.y * src.complNum.y;

    double result = (re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.complNum.x * b * c, a.complNum.y * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
GPUComplex GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.complNum.x*b, a.complNum.y*b);
   return result;

}

/*
 * Return the complex number a += b * c  
 */
void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.complNum.x * c.complNum.x - b.complNum.y*c.complNum.y ;
    double im_this = b.complNum.x * c.complNum.y + b.complNum.y*c.complNum.x ;

    GPUComplex mult_result(re_this, im_this);

    a.complNum.x += mult_result.complNum.x;
    a.complNum.y += mult_result.complNum.y;
}

/*
 * Return the complex number a -= b * c  
 */
void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.complNum.x * c.complNum.x - b.complNum.y*c.complNum.y ;
    double im_this = b.complNum.x * c.complNum.y + b.complNum.y*c.complNum.x ;

    GPUComplex mult_result(re_this, im_this);

    a.complNum.x -= mult_result.complNum.x;
    a.complNum.y -= mult_result.complNum.y;
}


GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    GPUComplex result(a - src.complNum.x, 0 - src.complNum.y);
    return result;
}

GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.complNum.x, 0 + src.complNum.y);
    return result;
}

double GPUComplex_real( const GPUComplex& src) {
    return src.complNum.x;
}

double GPUComplex_imag( const GPUComplex& src) {
    return src.complNum.y;
}

void gppKernelGPU( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex* aqsmtemp, GPUComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index);

void till_nvbandKernel(GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxtemp, int *inv_igp_index, int *indinv, GPUComplex *wtilde_array, double *wx_array, GPUComplex *I_eps_array, int ncouls, int nvband, int ngpown, int nstart, int nend, double *d_vcoul);

#endif
