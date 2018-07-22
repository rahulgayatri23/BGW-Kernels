/*
CustomComplex class that represents a complex class comprised of double's to represent a real and imaginary parts.
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


class CustomComplex {

    private : 
    double re;
    double im;

    public:
    explicit CustomComplex () {
        re = 0.00;
        im = 0.00;
    }


    explicit CustomComplex(const double& x, const double& y) {
        re = x;
        im = y;
    }

    CustomComplex(const CustomComplex& src) {
        re = src.re;
        im = src.im;
    }

    CustomComplex& operator =(const CustomComplex& src) {
        re = src.re;
        im = src.im;

        return *this;
    }

    CustomComplex& operator +=(const CustomComplex& src) {
        re = src.re + this->re;
        im = src.im + this->im;

        return *this;
    }

    CustomComplex& operator -=(const CustomComplex& src) {
        re = src.re - this->re;
        im = src.im - this->im;

        return *this;
    }

    CustomComplex& operator -() {
        re = -this->re;
        im = -this->im;

        return *this;
    }

    CustomComplex& operator ~() {
        return *this;
    }

    void print() const {
        printf("( %f, %f) ", this->re, this->im);
        printf("\n");
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

    friend inline CustomComplex operator *(const CustomComplex &a, const CustomComplex &b) {
        double x_this = a.re * b.re - a.im*b.im ;
        double y_this = a.re * b.im + a.im*b.re ;
        CustomComplex result(x_this, y_this);
        return (result);
    }

    friend inline CustomComplex operator *(const CustomComplex &a, const double &b) {
       CustomComplex result(a.re*b, a.im*b);
       return result;
    }

    friend inline CustomComplex operator -(const double &a, CustomComplex& src) {
        CustomComplex result(a - src.re, 0 - src.im);
        return result;
    }

    friend inline CustomComplex operator +(const double &a, CustomComplex& src) {
        CustomComplex result(a + src.re, src.im);
        return result;
    }

    friend inline CustomComplex CustomComplex_conj(const CustomComplex& src) ;
    friend inline double CustomComplex_abs(const CustomComplex& src) ;
    friend inline double CustomComplex_real( const CustomComplex& src) ;
    friend inline double CustomComplex_imag( const CustomComplex& src) ;
};

    inline CustomComplex CustomComplex_conj(const CustomComplex& src) ;
    inline double CustomComplex_abs(const CustomComplex& src) ;

//Inline functions have to be defined in the same file as the declaration

/*
 * Return the conjugate of a complex number 
 1flop
 */
CustomComplex CustomComplex_conj(const CustomComplex& src) {

    double re_this = src.re;
    double im_this = -1 * src.im;

    CustomComplex result(re_this, im_this);
    return result;

}

/*
 * Return the absolute of a complex number 
 */
double CustomComplex_abs(const CustomComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = sqrt(re_this+im_this);
    return result;
}

/*
 * Return the real part of a complex number 
 */
double CustomComplex_real( const CustomComplex& src) {
    return src.re;
}

/*
 * Return the imaginary part of a complex number 
 */
double CustomComplex_imag( const CustomComplex& src) {
    return src.im;
}

#endif
