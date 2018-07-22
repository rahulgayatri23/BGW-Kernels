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

template<class real, class imag>
class CustomComplex {

    private : 
    real re;
    imag im;

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

template<class real, class imag>
    friend inline CustomComplex<real,imag> operator *(const CustomComplex<real,imag> &a, const CustomComplex<real,imag> &b) {
        real x_this = a.re * b.re - a.im*b.im ;
        imag y_this = a.re * b.im + a.im*b.re ;
        CustomComplex<real,imag> result(x_this, y_this);
        return (result);
    }

template<class real, class imag>
    friend inline CustomComplex<real,imag> operator *(const CustomComplex<real,imag> &a, const double &b) {
       CustomComplex<real,imag> result(a.re*b, a.im*b);
       return result;
    }

template<class real, class imag>
    friend inline CustomComplex<real,imag> operator -(const double &a, CustomComplex<real,imag>& src) {
        CustomComplex<real,imag> result(a - src.re, 0 - src.im);
        return result;
    }

template<class real, class imag>
    friend inline CustomComplex<real,imag> operator +(const double &a, CustomComplex<real,imag>& src) {
        CustomComplex<real,imag> result(a + src.re, src.im);
        return result;
    }

template<class real, class imag>
    friend inline CustomComplex<real,imag> CustomComplex_conj(const CustomComplex<real,imag>& src) ;
template<class real, class imag>
    friend inline double CustomComplex_abs(const CustomComplex<real,imag>& src) ;
template<class real, class imag>
    friend inline double CustomComplex_real( const CustomComplex<real,imag>& src) ;
template<class real, class imag>
    friend inline double CustomComplex_imag( const CustomComplex<real,imag>& src) ;
};

/*
 * Return the conjugate of a complex number 
 1flop
 */
template<class real, class imag>
inline CustomComplex<real, imag> CustomComplex_conj(const CustomComplex<real,imag>& src) {

    real re_this = src.re;
    imag im_this = -1 * src.im;

    CustomComplex<real,imag> result(re_this, im_this);
    return result;

}

/*
 * Return the absolute of a complex number 
 */
template<class real, class imag>
inline double CustomComplex_abs(const CustomComplex<real,imag>& src) {
    real re_this = src.re * src.re;
    imag im_this = src.im * src.im;

    real result = sqrt(re_this+im_this);
    return result;
}

/*
 * Return the real part of a complex number 
 */
template<class real, class imag>
inline double CustomComplex_real( const CustomComplex<real,imag>& src) {
    return src.re;
}

/*
 * Return the imaginary part of a complex number 
 */
template<class real, class imag>
inline double CustomComplex_imag( const CustomComplex<real,imag>& src) {
    return src.im;
}

#endif
