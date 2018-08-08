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
#define nend 6

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

//inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, double  *aqsmtemp, double *aqsntemp, double *I_eps_array, double achstemp,  int* indinv, int ngpown, double* vcoul)
//{
//    double to1 = 1e-6;
//    double schstemp(0.0, 0.0);;
//
//    for(int my_igp = 0; my_igp< ngpown; my_igp++)
//    {
//        double schs(0.0, 0.0);
//        double matngmatmgp(0.0, 0.0);
//        double matngpmatmg(0.0, 0.0);
//        double halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
//        int indigp = inv_igp_index[my_igp];
//        int igp = indinv[indigp];
//        if(indigp == ncouls)
//            igp = ncouls-1;
//
//        if(!(igp > ncouls || igp < 0)){
//
//            double mygpvar2, mygpvar1;
//            mygpvar1 = double_conj(aqsmtemp[n1*ncouls+igp]);
//            mygpvar2 = aqsntemp[n1*ncouls+igp];
//
//
//
//            schs = I_eps_array[my_igp*ncouls+igp];
//            matngmatmgp = double_product(mygpvar1, aqsntemp[n1*ncouls+igp]);
//
//
//            if(double_abs(schs) > to1)
//                double_fma(schstemp, matngmatmgp, schs);
//            }
//            else 
//            {
//                for(int ig=1; ig<ncouls; ++ig)
//                {
//                    double mult_result(double_product(I_eps_array[my_igp*ncouls+ig] , mygpvar1));
//                    double_fms(schstemp,aqsntemp[n1*ncouls+igp], mult_result); 
//                }
//            }
//
//        schstemp = double_mult(schstemp, vcoul[igp], 0.5);
//        achstemp += schstemp;
//    }
//}
//
//inline void flagOCC_solver(double wxt, double *wtilde_array, int my_igp, int n1, double *aqsmtemp, double *aqsntemp, double *I_eps_array, double &ssxt, double &scht,int ncouls, int igp, int number_bands, int ngpown)
//{
//    double expr0(0.00, 0.00);
//    double expr(0.5, 0.5);
//    double matngmatmgp(0.0, 0.0);
//    double matngpmatmg(0.0, 0.0);
//
//    for(int ig=0; ig<ncouls; ++ig)
//    {
//        double wtilde = wtilde_array[my_igp*ncouls+ig];
//        double wtilde2 = double_square(wtilde);
//        double Omega2 = double_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
//        double mygpvar1 = double_conj(aqsmtemp[n1*ncouls+igp]);
//        double mygpvar2 = aqsmtemp[n1*ncouls+igp];
//        double matngmatmgp = double_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
//        if(ig != igp) matngpmatmg = double_product(double_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);
//
//        double delw2, scha_mult, ssxcutoff;
//        double to1 = 1e-6;
//        double sexcut = 4.0;
//        double gamma = 0.5;
//        double limitone = 1.0/(to1*4.0);
//        double limittwo = pow(0.5,2);
//        double sch, ssx;
//    
//        double wdiff = doubleMinusdouble(wxt , wtilde);
//    
//        double cden = wdiff;
//        double rden = 1/double_real(double_product(cden , double_conj(cden)));
//        double delw = double_mult(double_product(wtilde , double_conj(cden)) , rden);
//        double delwr = double_real(double_product(delw , double_conj(delw)));
//        double wdiffr = double_real(double_product(wdiff , double_conj(wdiff)));
//    
//        if((wdiffr > limittwo) && (delwr < limitone))
//        {
//            sch = double_product(delw , I_eps_array[my_igp*ngpown+ig]);
//            double cden = std::pow(wxt,2);
//            rden = std::pow(cden,2);
//            rden = 1.00 / rden;
//            ssx = double_mult(Omega2 , cden , rden);
//        }
//        else if (delwr > to1)
//        {
//            sch = expr0;
//            cden = double_mult(double_product(wtilde2, doublePlusdouble((double)0.50, delw)), 4.00);
//            rden = double_real(double_product(cden , double_conj(cden)));
//            rden = 1.00/rden;
//            ssx = double_product(double_product(-Omega2 , double_conj(cden)), double_mult(delw, rden));
//        }
//        else
//        {
//            sch = expr0;
//            ssx = expr0;
//        }
//    
//        ssxcutoff = double_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
//        if((double_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;
//
//        ssxt += double_product(matngmatmgp , ssx);
//        scht += double_product(matngmatmgp , sch);
//    }
//}
//
//void gppKernelCPU( double *wtilde_array, double *aqsntemp, double *I_eps_array, int ncouls, double wxt, double& achtemp_re_iw, double& achtemp_im_iw, int my_igp, double mygpvar1, int n1, double vcoul_igp)
//{
//    double scht(0.00, 0.00);
//    for(int ig = 0; ig<ncouls; ++ig)
//    {
//
//        double wdiff = doubleMinusdouble(wxt , wtilde_array[my_igp*ncouls+ig]);
//        double rden = double_real(double_product(wdiff, double_conj(wdiff)));
//        rden = 1/rden;
//        double delw = double_mult(double_product(wtilde_array[my_igp*ncouls+ig] , double_conj(wdiff)), rden); 
//        
//        scht += double_mult(double_product(double_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), double_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5);
//    }
//    achtemp_re_iw += double_real( double_mult(scht , vcoul_igp));
//    achtemp_im_iw += double_imag( double_mult(scht , vcoul_igp));
//}

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
