#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <ctime>
#include <chrono>
#include<inttypes.h>

#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define nstart 0
#define nend 3

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

void gppKernelGPU( double *wtilde_array, double *aqsntemp, double* aqsmtemp, double *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp, double *vcoul, int* indinv, int* inv_igp_index, int stride);


inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed. - Rahul - commented the below deviceSynchronize
//    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}


using namespace std;

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
    
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];

    double *acht_n1_loc = new double[number_bands];
    double *achtemp = new double[(nend-nstart)];
    double *aqsmtemp = new double[number_bands*ncouls];
    double *aqsntemp = new double[number_bands*ncouls];
    double *I_eps_array = new double[ngpown*ncouls];
    double *wtilde_array = new double[ngpown*ncouls];
    double *ssx_array = new double[(nend-nstart)];
    double *ssxa = new double[ncouls];
    double achstemp;

    double *wx_array = new double[(nend-nstart)];
    double *vcoul = new double[ncouls];

    printf("Executing CUDA version of the Kernel stride = %d\n", stride);
//Data Structures on Device
    double *d_wtilde_array, *d_aqsntemp, *d_aqsmtemp, *d_I_eps_array, *d_asxtemp;
    double *d_achtemp, *d_vcoul, *d_wx_array;
    int *d_inv_igp_index, *d_indinv;

    CudaSafeCall(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_wx_array, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_indinv, (ncouls+1)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_asxtemp, (nend-nstart)*sizeof(double)));

    double occ=1.0;
    bool flag_occ;

   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = 0.5;
           aqsntemp[i*ncouls+j] = 0.5;
       }

   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = 0.5;
           wtilde_array[i*ncouls+j] = 0.5;
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

    CudaSafeCall(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(double), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(double), cudaMemcpyHostToDevice));
    mem_alloc += 2*ngpown*ncouls*sizeof(double);

    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(double), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(double), cudaMemcpyHostToDevice));
    mem_alloc += 2*number_bands*ncouls*sizeof(double);

    CudaSafeCall(cudaMemcpy(d_indinv, indinv, (ncouls+1)*sizeof(int), cudaMemcpyHostToDevice));
    mem_alloc += ncouls*sizeof(int);

    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    mem_alloc += ngpown*sizeof(int);

    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    mem_alloc += ncouls*sizeof(double);

    CudaSafeCall(cudaMemcpy(d_wx_array, wx_array, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_achtemp, achtemp, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));

    mem_alloc += 3*3*sizeof(double);

    mem_alloc /= (1024*1024*1024);

    printf("mem_alloc = %f GBs\n", mem_alloc);
//Start Kernel 
    auto start_kernelTiming = std::chrono::high_resolution_clock::now();

//    till_nvbandKernel(d_aqsmtemp, d_aqsntemp, d_asxtemp, d_inv_igp_index, d_indinv, d_wtilde_array, d_wx_array, d_I_eps_array, ncouls, nvband, ngpown, d_vcoul);

    gppKernelGPU( d_wtilde_array, d_aqsntemp, d_aqsmtemp, d_I_eps_array, ncouls, ngpown, number_bands, d_wx_array, d_achtemp, d_vcoul, d_indinv, d_inv_igp_index, stride);

    cudaDeviceSynchronize();
    std::chrono::duration<double> elapsed_kernelTiming = std::chrono::high_resolution_clock::now() - start_kernelTiming;

//Start memcpyToHost 
    CudaSafeCall(cudaMemcpy(achtemp, d_achtemp, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));

    printf(" \n Cuda Kernel Final achtemp\n");
    cout << "achtemp[0] = " << achtemp[0] << endl;

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;

    cout << "********** Kernel Time Taken **********= " << elapsed_kernelTiming.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    cudaFree(d_wtilde_array);
    cudaFree(d_aqsntemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_asxtemp);
    cudaFree(d_I_eps_array);
    cudaFree(d_achtemp);
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
