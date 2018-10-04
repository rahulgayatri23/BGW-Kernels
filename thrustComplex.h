#ifndef __CustomComplex_thrust
#define __CustomComplex_thrust

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/complex.h>

using namespace std;

//Using Thurst Complex 
typedef double REAL;
using CustomComplex = thrust::complex<REAL>;

//GPP Function definition
//template<class T>
void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im);

//FF Function definition
//template<class T>
inline void schDttt_corKernel1(CustomComplex &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

//template<class T>
inline void schDttt_corKernel2(CustomComplex &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

//template<class T>
void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, CustomComplex *aqsntemp, CustomComplex *aqsmtemp, CustomComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im);

//template<class T>
void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im);

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, CustomComplex *achDtemp_corb);

//TestUVM functions
void cudaKernel(int *a, int N, int M);

#endif

