#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>

#define nstart 0
#define nend 3

__device__ void ncoulsKernel(double wx_array_iw, double wtilde_array, double aqsmtemp, double aqsntemp, double I_eps_array, double vcoul, double &achtemp)
{
    double wdiff = wx_array_iw - wtilde_array;
    double delw = wtilde_array * wdiff * 1/(wdiff * wdiff); 
    double sch_array = aqsmtemp * aqsntemp * delw * I_eps_array * 0.5*vcoul; 
    achtemp += sch_array;
}

__global__  void NumBandNgpown_kernel( double *wtilde_array, double *aqsntemp, double* aqsmtemp, double *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double* achtemp, double* vcoul, int* indinv, int* inv_igp_index, int numThreadsPerBlock, int stride)
{
    int n1 = blockIdx.x;
    int my_igp = blockIdx.y;

    if((n1 < number_bands ) && (my_igp < ngpown) )
    {
        int loopOverncouls = 1, leftOverncouls = 0;
        if(ncouls > numThreadsPerBlock)
        {
            loopOverncouls = ncouls / numThreadsPerBlock;
            leftOverncouls = ncouls % numThreadsPerBlock;
        }

        double achtemp_loc[nend-nstart];
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        for(int iw = nstart; iw < nend; ++iw)
            achtemp_loc[iw] = 0.00;

        if(stride == 0)
        {
            for( int x = 0; x < loopOverncouls && threadIdx.x < numThreadsPerBlock ; ++x)
            { 
                int ig = x*numThreadsPerBlock + threadIdx.x;
                if(ig < ncouls)
                    for(int iw = nstart; iw < nend; ++iw)
                        ncoulsKernel(wx_array[iw], wtilde_array[my_igp*ncouls + ig], aqsmtemp[n1*ncouls + igp], aqsntemp[n1*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[ig], achtemp_loc[iw]);
            }
            if(leftOverncouls)
            {
                int ig = loopOverncouls*numThreadsPerBlock + threadIdx.x;
                if(ig < ncouls)
                    for(int iw = nstart; iw < nend; ++iw)
                        ncoulsKernel(wx_array[iw], wtilde_array[my_igp*ncouls + ig], aqsmtemp[n1*ncouls + igp], aqsntemp[n1*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[ig], achtemp_loc[iw]);
            }
          }
          else
          {
              for(int igmin = 0; igmin < stride; ++igmin)
              { 
                  for( int x = 0; x < loopOverncouls/stride && threadIdx.x < numThreadsPerBlock ; ++x)
                  {
                      int ig = (x*numThreadsPerBlock + threadIdx.x) * stride + igmin ;
                      if(ig < ncouls)
                        for(int iw = nstart; iw < nend; ++iw)
                            ncoulsKernel(wx_array[iw], wtilde_array[my_igp*ncouls + ig], aqsmtemp[n1*ncouls + igp], aqsntemp[n1*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[ig], achtemp_loc[iw]);
                  }
              }
              if(leftOverncouls)
              {
                  for(int igmin = 0; igmin < stride; ++igmin)
                  {
                      int ig = loopOverncouls*numThreadsPerBlock + threadIdx.x*stride + igmin;
                      if(ig < ncouls)
                        for(int iw = nstart; iw < nend; ++iw)
                            ncoulsKernel(wx_array[iw], wtilde_array[my_igp*ncouls + ig], aqsmtemp[n1*ncouls + igp], aqsntemp[n1*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[ig], achtemp_loc[iw]);
                  }
              }
          }
            for(int iw = nstart; iw < nend; ++iw)
                atomicAdd(&achtemp[iw] , achtemp_loc[iw] );
    }
}

void gppKernelGPU( double *wtilde_array, double *aqsntemp, double* aqsmtemp, double *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp, double *vcoul, int* indinv, int* inv_igp_index, int stride)
{
    dim3 numBlocks(number_bands, ngpown);
    int numThreadsPerBlock = 32;

    printf("Launching a double dimension grid with numBlocks = (%d, %d) and %d threadsPerBlock \n", number_bands, ngpown, numThreadsPerBlock);

    NumBandNgpown_kernel  <<< numBlocks, numThreadsPerBlock >>> ( wtilde_array, aqsntemp, aqsmtemp, I_eps_array, ncouls, ngpown, number_bands, wx_array, achtemp, vcoul, indinv, inv_igp_index, numThreadsPerBlock, stride);
}
