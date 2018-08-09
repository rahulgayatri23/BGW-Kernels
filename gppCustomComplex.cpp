#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>
#include<inttypes.h>

#include "CustomComplex.h"

using namespace std;

CustomComplex** allocateMemGPU(size_t size)
{
    void **d_src;
    if(cudaMalloc((void**) &d_src, size) != cudaSuccess)
    {
        return NULL;
    }

    return (CustomComplex**) d_src;
}

int main(int argc, char** argv)
{

    if (argc != 6)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> <stride> " << endl;
        exit (0);
    }

    printf("********Executing Cuda version of the Kernel*********\n");

    auto start_totalTime = std::chrono::high_resolution_clock::now();
    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int nodes_per_group = atoi(argv[4]);
    const int stride = atoi(argv[5]);
    const int npes = 1; 
    const int ngpown = ncouls / (nodes_per_group * npes); 
    const double e_lk = 10;
    const double dw = 1;

    double to1 = 1e-6, \
    gamma = 0.5, \
    sexcut = 4.0;
    double limitone = 1.0/(to1*4.0), \
    limittwo = pow(0.5,2);
    const double e_n1kq= 6.0; 

    //Printing out the params passed.
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    //ALLOCATE statements from fortran gppkernel.
    
   
    CustomComplex expr0(0.00, 0.00);
    CustomComplex expr(0.5, 0.5);

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];

    CustomComplex *acht_n1_loc = new CustomComplex[number_bands];
    CustomComplex *achtemp = new CustomComplex[(nend-nstart)];
    CustomComplex *aqsmtemp = new CustomComplex[number_bands*ncouls];
    CustomComplex *aqsntemp = new CustomComplex[number_bands*ncouls];
    CustomComplex *I_eps_array = new CustomComplex[ngpown*ncouls];
    CustomComplex *wtilde_array = new CustomComplex[ngpown*ncouls];
    CustomComplex *ssx_array = new CustomComplex[(nend-nstart)];
    CustomComplex *ssxa = new CustomComplex[ncouls];
    CustomComplex achstemp;

    double *achtemp_re = new double[(nend-nstart)];
    double *wx_array = new double[(nend-nstart)];
    double *achtemp_im = new double[(nend-nstart)];
    double *vcoul = new double[ncouls];

    printf("Executing CUDA version of the Kernel stride = %d\n", stride);
//Data Structures on Device
    CustomComplex *d_wtilde_array, *d_aqsntemp, *d_aqsmtemp, *d_I_eps_array, *d_asxtemp;
    double *d_achtemp_re, *d_achtemp_im, *d_vcoul, *d_wx_array;
    int *d_inv_igp_index, *d_indinv;

    CudaSafeCall(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_re, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_im, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_wx_array, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_indinv, (ncouls+1)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_asxtemp, (nend-nstart)*sizeof(double)));

   double occ=1.0;
   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j].x = 0.5;
           aqsmtemp[i*ncouls+j].y = 0.5;
           aqsntemp[i*ncouls+j].x = 0.5;
           aqsntemp[i*ncouls+j].y = 0.5;
       }

   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j].x = 0.5;
           I_eps_array[i*ncouls+j].y = 0.5;
           wtilde_array[i*ncouls+j].x = 0.5;
           wtilde_array[i*ncouls+j].y = 0.5;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;

    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0 ; ig<ncouls; ++ig)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

    for(int iw=nstart; iw<nend; ++iw)
    {
        wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
        if(wx_array[iw] < to1) wx_array[iw] = to1;
    }

    auto start_withDataMovement = std::chrono::high_resolution_clock::now();
    float mem_alloc = 0.00;

//Start memcpyToDevice 

    CudaSafeCall(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    mem_alloc += 2*ngpown*ncouls*sizeof(CustomComplex);

    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    mem_alloc += 2*number_bands*ncouls*sizeof(CustomComplex);

    CudaSafeCall(cudaMemcpy(d_indinv, indinv, (ncouls+1)*sizeof(int), cudaMemcpyHostToDevice));
    mem_alloc += ncouls*sizeof(int);

    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    mem_alloc += ngpown*sizeof(int);

    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    mem_alloc += ncouls*sizeof(double);

    CudaSafeCall(cudaMemcpy(d_wx_array, wx_array, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_achtemp_re, achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_achtemp_im, achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    mem_alloc += 3*3*sizeof(double);

    mem_alloc /= (1024*1024*1024);

    printf("memory footprint = %f GBs\n", mem_alloc);
//Start Kernel 
    auto start_kernelTiming = std::chrono::high_resolution_clock::now();

    till_nvbandKernel(d_aqsmtemp, d_aqsntemp, d_asxtemp, d_inv_igp_index, d_indinv, d_wtilde_array, d_wx_array, d_I_eps_array, ncouls, nvband, ngpown, d_vcoul);

    gppKernelGPU( d_wtilde_array, d_aqsntemp, d_aqsmtemp, d_I_eps_array, ncouls, ngpown, number_bands, d_wx_array, d_achtemp_re, d_achtemp_im, d_vcoul, d_indinv, d_inv_igp_index, stride);

    cudaDeviceSynchronize();
    std::chrono::duration<double> elapsed_kernelTiming = std::chrono::high_resolution_clock::now() - start_kernelTiming;

//Start memcpyToHost 
    CudaSafeCall(cudaMemcpy(achtemp_im, d_achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achtemp_re, d_achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));

    printf(" \n Cuda Kernel Final achtemp\n");
    for(int iw=nstart; iw<nend; ++iw)
    {
        achtemp[iw] = CustomComplex(achtemp_re[iw], achtemp_im[iw]);
//        achtemp[iw].print();
    }
    achtemp[0].print();

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;

    cout << "********** Kernel Time Taken **********= " << elapsed_kernelTiming.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    cudaFree(d_wtilde_array);
    cudaFree(d_aqsntemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_asxtemp);
    cudaFree(d_I_eps_array);
    cudaFree(d_achtemp_re);
    cudaFree(d_achtemp_im);
    cudaFree(d_vcoul);
    cudaFree(d_inv_igp_index);
    cudaFree(d_indinv);

    free(acht_n1_loc);
    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(vcoul);
    free(ssx_array);

    return 0;
}
