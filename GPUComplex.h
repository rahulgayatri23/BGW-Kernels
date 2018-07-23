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


#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

class GPUComplex : public double2{

    private : 

public:
//    double2 complNum;

explicit GPUComplex () {
    x = 0.00;
    y = 0.00;
}


explicit GPUComplex(const double& a, const double& b) {
    x = a;
    y = b;
}

GPUComplex(const GPUComplex& src) {
    x = src.x;
    y = src.y;
}

GPUComplex& operator =(const GPUComplex& src) {
    x = src.x;
    y = src.y;

    return *this;
}

GPUComplex& operator +=(const GPUComplex& src) {
    x = src.x + this->x;
    y = src.y + this->y;

    return *this;
}

GPUComplex& operator -=(const GPUComplex& src) {
    x = src.x - this->x;
    y = src.y - this->y;

    return *this;
}

GPUComplex& operator -() {
    x = -this->x;
    y = -this->y;

    return *this;
}

GPUComplex& operator ~() {
    return *this;
}

void print() const {
    printf("( %f, %f) ", this->x, this->y);
    printf("\n");
}

double abs(const GPUComplex& src) {

    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

    double result = (re_this+im_this);
    return result;
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

    GPUComplex GPUComplex_square(GPUComplex& src) ;
    GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    double GPUComplex_abs(const GPUComplex& src) ;
    GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
    GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
    GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    double GPUComplex_real( const GPUComplex& src) ;
    double GPUComplex_imag( const GPUComplex& src) ;
};

//Inline functions have to be defined in the same file as the declaration

/*
 * Return the square of a complex number 
 */
GPUComplex GPUComplex_square(GPUComplex& src) {
    double re_this = src.x ;
    double im_this = src.y ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
GPUComplex GPUComplex_conj(const GPUComplex& src) {

    double re_this = src.x;
    double im_this = -1 * src.y;

    GPUComplex result(re_this, im_this);
    return result;
}


/*
 * Return the product of 2 complex numbers 
 */
GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.x * b.x - a.y*b.y ;
    double im_this = a.x * b.y + a.y*b.x ;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

    double result = (re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.x * b * c, a.y * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
GPUComplex GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.x*b, a.y*b);
   return result;

}

/*
 * Return the complex number a += b * c  
 */
void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.x * c.x - b.y*c.y ;
    double im_this = b.x * c.y + b.y*c.x ;

    GPUComplex mult_result(re_this, im_this);

    a.x += mult_result.x;
    a.y += mult_result.y;
}

/*
 * Return the complex number a -= b * c  
 */
void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.x * c.x - b.y*c.y ;
    double im_this = b.x * c.y + b.y*c.x ;

    GPUComplex mult_result(re_this, im_this);

    a.x -= mult_result.x;
    a.y -= mult_result.y;
}


GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    GPUComplex result(a - src.x, 0 - src.y);
    return result;
}

GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.x, 0 + src.y);
    return result;
}

double GPUComplex_real( const GPUComplex& src) {
    return src.x;
}

double GPUComplex_imag( const GPUComplex& src) {
    return src.y;
}

void gppKernelGPU( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex* aqsmtemp, GPUComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index);

void till_nvbandKernel(GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxtemp, int *inv_igp_index, int *indinv, GPUComplex *wtilde_array, double *wx_array, GPUComplex *I_eps_array, int ncouls, int nvband, int ngpown, int nstart, int nend, double *d_vcoul);

#endif
