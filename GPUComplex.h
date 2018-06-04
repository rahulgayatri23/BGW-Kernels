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

public:
//    double re;
//    double im;

KOKKOS_INLINE_FUNCTION
explicit GPUComplex () {
    x = 0.00;
    y = 0.00;
}


KOKKOS_INLINE_FUNCTION
explicit GPUComplex(const double& a, const double& b) {
    x = a;
    y = b;
}

KOKKOS_INLINE_FUNCTION
GPUComplex(const GPUComplex& src) {
    x = src.x;
    y = src.y;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator =(const GPUComplex& src) {
    x = src.x;
    y = src.y;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator +=(const GPUComplex& src) {
    x = src.x + this->x;
    y = src.y + this->y;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator -=(const GPUComplex& src) {
    x = src.x - this->x;
    y = src.y - this->y;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator -() {
    x = -this->x;
    y = -this->y;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator ~() {
    return *this;
}

KOKKOS_INLINE_FUNCTION
void print() const {
    printf("( %f, %f) ", this->x, this->y);
    printf("\n");
}

KOKKOS_INLINE_FUNCTION
double abs(const GPUComplex& src) {

    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

//    double result = sqrt(re_this+im_this);
    double result = (re_this+im_this);

    return result;

}

KOKKOS_INLINE_FUNCTION
double get_real() const
{
    return this->x;
}

KOKKOS_INLINE_FUNCTION
double get_imag() const
{
    return this->y;
}

KOKKOS_INLINE_FUNCTION
void set_real(double val)
{
    this->x = val;
}

KOKKOS_INLINE_FUNCTION
void set_imag(double val) 
{
    this->y = val;
}


//    friend GPUComplex GPUComplex_square(GPUComplex& src) ;
//    friend GPUComplex GPUComplex_conj(const GPUComplex& src) ;
//    friend GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
//    friend double GPUComplex_abs(const GPUComplex& src) ;
//    friend GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
//    friend GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
//    friend void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
//    friend void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
//    friend GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
//    friend GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
//    friend double GPUComplex_real( const GPUComplex& src) ;
//    friend double GPUComplex_imag( const GPUComplex& src) ;
        
};
    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_square(GPUComplex& src) ;
    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    KOKKOS_INLINE_FUNCTION
    double GPUComplex_abs(const GPUComplex& src) ;
    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    KOKKOS_INLINE_FUNCTION
    void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    KOKKOS_INLINE_FUNCTION
    void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    KOKKOS_INLINE_FUNCTION
    GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;
    KOKKOS_INLINE_FUNCTION
    GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    KOKKOS_INLINE_FUNCTION
    double GPUComplex_real( const GPUComplex& src) ;
    KOKKOS_INLINE_FUNCTION
    double GPUComplex_imag( const GPUComplex& src) ;

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
//    double result = sqrt(re_this+im_this);
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

#endif
