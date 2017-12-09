#include "cudaComplex.h"
//
//#if CudaKernel
/*
 * Return the square of a complex number 
 */
__device__ const cuDoubleComplex d_cuDoubleComplex_square(cuDoubleComplex& src) {
    return make_cuDoubleComplex(src.x*src.x - src.y*src.y, 2*src.x*src.y);
}
//
///*
// * Return the conjugate of a complex number 
// */
__device__ inline cuDoubleComplex d_cuDoubleComplex_conj(cuDoubleComplex& src) {
return make_cuDoubleComplex(src.x, -src.y);
}


/*
 * Return the product of 2 complex numbers 
 */
__device__ const inline cuDoubleComplex d_cuDoubleComplex_product(const cuDoubleComplex& a, const cuDoubleComplex& b) {
    return make_cuDoubleComplex(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}


/*
 * Return the absolute of a complex number 
 */
__device__ inline double d_cuDoubleComplex_abs(const cuDoubleComplex& src) {
    return sqrt(src.x * src.x + src.y * src.y);
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
__device__ const inline cuDoubleComplex d_cuDoubleComplex_mult(cuDoubleComplex& a, double b, double c) {
    return make_cuDoubleComplex(a.x * b * c, a.y * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
__device__ const inline cuDoubleComplex d_cuDoubleComplex_mult(const cuDoubleComplex& a, double b) {
   return make_cuDoubleComplex(a.x*b, a.y*b);

}

/*
 * Return the complex number a += b * c  
 */
__device__ inline void d_cuDoubleComplex_fma(cuDoubleComplex& a, const cuDoubleComplex& b, const cuDoubleComplex& c) {
    a.x += b.x * c.x - b.y*c.y ;
    a.y += b.x * c.y + b.y*c.x ;
}

/*
 * Return the complex number a -= b * c  
 */
__device__ inline void d_cuDoubleComplex_fms(cuDoubleComplex& a, const cuDoubleComplex& b, const cuDoubleComplex& c) {
    a.x -= b.x * c.x - b.y*c.y ;
    a.x -= b.x * c.y + b.y*c.x ;
}


__device__ inline cuDoubleComplex d_doubleMinuscuDoubleComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a-src.x, -src.y);
}

__device__ inline cuDoubleComplex d_doubleMinuscuComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a - src.x, -src.y);
}

__device__ inline cuDoubleComplex d_doublePluscuComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a + src.x, src.y);
}


__device__ inline double d_cuDoubleComplex_real( const cuDoubleComplex& src) {
    return src.x;
}

__device__ inline double d_cuDoubleComplex_imag( const cuDoubleComplex& src) {
    return src.y;
}

__device__ inline void d_cuDoubleComplex_plusEquals( cuDoubleComplex& a, const cuDoubleComplex & b) {
    a.y += b.x;
    a.x += b.y;
}

__device__ void inline d_cuDoubleComplex_Equals( cuDoubleComplex& a, const cuDoubleComplex & b) {
    a.x = b.x;
    a.y = b.y;
}

__device__ __host__ void d_print( const cuDoubleComplex& a) {
    printf("( %f, %f) ", a.x, a.y);
    printf("\n");
}



__device__ void ncoulsKernel(cuDoubleComplex& mygpvar1, cuDoubleComplex& wdiff, cuDoubleComplex& aqsntemp_index, cuDoubleComplex& wtilde_array_index, cuDoubleComplex& I_eps_array_index, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc)
{
    double rden = 1/(wdiff.x*wdiff.x + wdiff.y*wdiff.y);

    achtemp_re_loc += d_cuDoubleComplex_product(d_cuDoubleComplex_product(mygpvar1, aqsntemp_index),\
        d_cuDoubleComplex_product(d_cuDoubleComplex_mult(d_cuDoubleComplex_product(wtilde_array_index, d_cuDoubleComplex_conj(wdiff)), rden), I_eps_array_index)).x * 0.5 * vcoul_igp;
    achtemp_im_loc += d_cuDoubleComplex_product(d_cuDoubleComplex_product(mygpvar1, aqsntemp_index),\
        d_cuDoubleComplex_product(d_cuDoubleComplex_mult(d_cuDoubleComplex_product(wtilde_array_index, d_cuDoubleComplex_conj(wdiff)), rden), I_eps_array_index)).y * 0.5 * vcoul_igp;
}

__global__  void cudaBGWKernel_ncouls_ngpown( cuDoubleComplex *wtilde_array, cuDoubleComplex *aqsntemp, cuDoubleComplex* aqsmtemp, cuDoubleComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double* achtemp_re, double* achtemp_im, double* vcoul, int nstart, int nend, int* indinv, int* inv_igp_index, int numThreadsPerBlock)
{
    int n1 = blockIdx.x ;
    int my_igp = blockIdx.y;

    if(n1 < number_bands && my_igp < ngpown)
    {
        int loopOverncouls = 1, leftOverncouls = 0, \
            threadsPerBlock = numThreadsPerBlock;

        if(ncouls > threadsPerBlock)
        {
            loopOverncouls = ncouls / threadsPerBlock;
            leftOverncouls = ncouls % threadsPerBlock;
        }

        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];

        for(int iw = nstart; iw < nend; ++iw)
        {
            double achtemp_re_loc = 0.00, achtemp_im_loc = 0.00;

            for( int x = 0; x < loopOverncouls && threadIdx.x < threadsPerBlock ; ++x)
            {
                int ig = x*threadsPerBlock + threadIdx.x;

                if(ig < ncouls)
                { 
                    cuDoubleComplex mygpvar1 = d_cuDoubleComplex_conj(aqsmtemp[n1*ncouls +igp]);
                    cuDoubleComplex wdiff = d_doubleMinuscuComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                    ncoulsKernel(mygpvar1, wdiff, aqsntemp[n1*ncouls+ig], wtilde_array[my_igp*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[igp], achtemp_re_loc, achtemp_im_loc);
                } //ncouls
            }

            if(leftOverncouls)
            {
                int ig = loopOverncouls*threadsPerBlock + threadIdx.x;
                if(ig < ncouls)
                {
                    cuDoubleComplex mygpvar1 = d_cuDoubleComplex_conj(aqsmtemp[n1*ncouls +igp]);
                    cuDoubleComplex wdiff = d_doubleMinuscuComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                    ncoulsKernel(mygpvar1, wdiff, aqsntemp[n1*ncouls+ig], wtilde_array[my_igp*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[igp], achtemp_re_loc, achtemp_im_loc);
                } //ncouls
            }

            atomicAdd(&achtemp_re[iw] , achtemp_re_loc);
            atomicAdd(&achtemp_im[iw] , achtemp_im_loc );
        } // iw
    }
}

__global__ void d_flagOCC_solver(double *wx_array, cuDoubleComplex *wtilde_array, cuDoubleComplex* asxtemp, cuDoubleComplex *aqsmtemp, cuDoubleComplex *aqsntemp, cuDoubleComplex *I_eps_array, int* inv_igp_index, int* indinv, int ncouls, int nvband, int ngpown, int nstart, int nend, double* vcoul)
{
    int n1 = blockIdx.x ;
    int my_igp = blockIdx.y;

    if(n1 < nvband && my_igp < ngpown)
    {
        int loopOverncouls = 1, \
            threadsPerBlock = 128;

        if(ncouls > threadsPerBlock)
            loopOverncouls = ncouls / threadsPerBlock;

        for(int iw = nstart; iw < nend; ++iw)
        {
            double wxt = wx_array[iw];
            for( int x = 0; x < loopOverncouls && threadIdx.x < threadsPerBlock ; ++x)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                cuDoubleComplex ssxt = make_cuDoubleComplex(0.00, 0.00);
                cuDoubleComplex scht = make_cuDoubleComplex(0.00, 0.00);
                        
                cuDoubleComplex expr = make_cuDoubleComplex(0.50, 0.50);
                cuDoubleComplex expr0 = make_cuDoubleComplex(0.00, 0.00);
                cuDoubleComplex matngmatmgp = make_cuDoubleComplex(0.00, 0.00);
                cuDoubleComplex matngpmatmg = make_cuDoubleComplex(0.00, 0.00);
        
                for(int ig=0; ig<ncouls; ++ig)
                {
                    cuDoubleComplex wtilde = wtilde_array[my_igp*ncouls+ig];
                    cuDoubleComplex wtilde2 = d_cuDoubleComplex_square(wtilde);
                    cuDoubleComplex Omega2 = d_cuDoubleComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
                    cuDoubleComplex mygpvar1 = d_cuDoubleComplex_conj(aqsmtemp[n1*ncouls+igp]);
                    cuDoubleComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
                    cuDoubleComplex matngmatmgp = d_cuDoubleComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
                    if(ig != igp) matngpmatmg = d_cuDoubleComplex_product(d_cuDoubleComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);
        
                    double to1 = 1e-6;
                    double sexcut = 4.0;
                    double limitone = 1.0/(to1*4.0);
                    double limittwo = pow(0.5,2);
                    cuDoubleComplex ssx;
                
                    cuDoubleComplex wdiff = d_doubleMinuscuDoubleComplex(wxt , wtilde);
                
                    cuDoubleComplex cden = wdiff;
                    double rden = 1/d_cuDoubleComplex_real(d_cuDoubleComplex_product(cden , d_cuDoubleComplex_conj(cden)));
                    cuDoubleComplex delw = d_cuDoubleComplex_mult(d_cuDoubleComplex_product(wtilde , d_cuDoubleComplex_conj(cden)) , rden);
                    double delwr = d_cuDoubleComplex_real(d_cuDoubleComplex_product(delw , d_cuDoubleComplex_conj(delw)));
                    double wdiffr = d_cuDoubleComplex_real(d_cuDoubleComplex_product(wdiff , d_cuDoubleComplex_conj(wdiff)));
                
                    if((wdiffr > limittwo) && (delwr < limitone))
                    {
                       double cden = std::pow(wxt,2);
                        rden = std::pow(cden,2);
                        rden = 1.00 / rden;
                        ssx = d_cuDoubleComplex_mult(Omega2 , cden , rden);
                    }
                    else if (delwr > to1)
                    {
                        cden = d_cuDoubleComplex_mult(d_cuDoubleComplex_product(wtilde2, d_doublePluscuComplex((double)0.50, delw)), 4.00);
                       rden = d_cuDoubleComplex_real(d_cuDoubleComplex_product(cden , d_cuDoubleComplex_conj(cden)));
                        rden = 1.00/rden;
                        ssx = d_cuDoubleComplex_product(d_cuDoubleComplex_product(make_cuDoubleComplex(-Omega2.x, -Omega2.y) , d_cuDoubleComplex_conj(cden)), d_cuDoubleComplex_mult(delw, rden));
                    }
                    else
                    {
                        ssx = expr0;
                    }
                
                    double ssxcutoff = d_cuDoubleComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
                    if((d_cuDoubleComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;
        
                    d_cuDoubleComplex_plusEquals(ssxt, d_cuDoubleComplex_product(matngmatmgp , ssxt));
                    d_cuDoubleComplex_plusEquals(scht, d_cuDoubleComplex_product(matngmatmgp , scht));
                }
                        d_cuDoubleComplex_plusEquals(asxtemp[iw] , d_cuDoubleComplex_mult(ssxt , vcoul[igp]));
            }
        }
    }
}

void gppKernelGPU( cuDoubleComplex *wtilde_array, cuDoubleComplex *aqsntemp, cuDoubleComplex* aqsmtemp, cuDoubleComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index)
{
    printf("gppKernelGPU for cuComplex class\n");
    dim3 numBlocks(number_bands, ngpown);
    int numThreadsPerBlock = 32;
    printf("launching 2 dimension grid with (number_bands, ngpown) dime and then calling ncouls loop by threads inside \n");

    cudaBGWKernel_ncouls_ngpown <<< numBlocks, numThreadsPerBlock>>> ( wtilde_array, aqsntemp, aqsmtemp, I_eps_array, ncouls, ngpown, number_bands, wx_array, achtemp_re, achtemp_im, vcoul, nstart, nend, indinv, inv_igp_index, numThreadsPerBlock);
}


void till_nvbandKernel(cuDoubleComplex* aqsmtemp, cuDoubleComplex* aqsntemp, cuDoubleComplex* asxtemp, int *inv_igp_index, int *indinv, cuDoubleComplex *wtilde_array, double *wx_array, cuDoubleComplex *I_eps_array, int ncouls, int nvband, int ngpown, int nstart, int nend, double* vcoul)
{
    dim3 numBlocks(nvband, ngpown);
    int numThreadsPerBlock = ncouls;

    d_flagOCC_solver<<< numBlocks, numThreadsPerBlock>>>(wx_array, wtilde_array, asxtemp, aqsmtemp, aqsntemp, I_eps_array, inv_igp_index, indinv, ncouls, nvband, ngpown, nstart, nend, vcoul);

}
