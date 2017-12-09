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

class GPUComplex : public double2{

    private : 
    double re;
    double im;

public:
#pragma omp declare target
explicit GPUComplex () {
    re = 0.00;
    im = 0.00;
}


explicit GPUComplex(const double& x, const double& y) {
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
    printf("\( %f, %f) ", this->re, this->im);
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


    friend inline const GPUComplex GPUComplex_square(GPUComplex& src) ;
    friend inline const GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    friend inline const GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    friend inline const double GPUComplex_abs(const GPUComplex& src) ;
    friend inline const GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
    friend inline const GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    friend inline const void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend inline const void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend inline GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
    friend inline const GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    friend inline double GPUComplex_real( const GPUComplex& src) ;
    friend inline double GPUComplex_imag( const GPUComplex& src) ;
#pragma omp end declare target
        
};
#pragma omp declare target
    inline const GPUComplex GPUComplex_square(GPUComplex& src) ;
    inline const GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    inline const GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    inline const double GPUComplex_abs(const GPUComplex& src) ;
    inline const GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
    inline const GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    inline const void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    inline const void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    inline GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
    inline const GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    inline double GPUComplex_real( const GPUComplex& src) ;
    inline double GPUComplex_imag( const GPUComplex& src) ;
#pragma omp end declare target

//Inline functions have to be defined in the same file as the declaration

/*
 * Return the square of a complex number 
 */
const GPUComplex GPUComplex_square(GPUComplex& src) {
    double re_this = src.re ;
    double im_this = src.im ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
const GPUComplex GPUComplex_conj(const GPUComplex& src) {

double re_this = src.re;
double im_this = -1 * src.im;

GPUComplex result(re_this, im_this);
return result;

}


/*
 * Return the product of 2 complex numbers 
 */
const GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.re * b.re - a.im*b.im ;
    double im_this = a.re * b.im + a.im*b.re ;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
const double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = (re_this+im_this);
//    double result = sqrt(re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
const GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.re * b * c, a.im * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
const GPUComplex GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.re*b, a.im*b);
   return result;

}

/*
 * Return the complex number a += b * c  
 */
const void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re += mult_result.re;
    a.im += mult_result.im;
}

/*
 * Return the complex number a -= b * c  
 */
const void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
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

const GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.re, 0 + src.im);
    return result;
}

double GPUComplex_real( const GPUComplex& src) {
    return src.re;
}

double GPUComplex_imag( const GPUComplex& src) {
    return src.im;
}

#endif
