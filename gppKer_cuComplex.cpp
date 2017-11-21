//#include <iostream>
//#include <cstdlib>
//#include <memory>
//
//#include <iomanip>
//#include <cmath>
//#include <complex>
//#include <omp.h>
//#include <ctime>
//#include <chrono>
//#include <cuComplex.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include "cuComplex.h"

using namespace std;
int debug = 0;

inline void flagOCC_solver(double wxt, cuDoubleComplex *wtilde_array, int my_igp, int n1, cuDoubleComplex *aqsmtemp, cuDoubleComplex *aqsntemp, cuDoubleComplex *I_eps_array, cuDoubleComplex &ssxt, cuDoubleComplex &scht,int ncouls, int igp, int number_bands, int ngpown)
{
    cuDoubleComplex expr = make_cuDoubleComplex(0.50, 0.50);
    cuDoubleComplex expr0 = make_cuDoubleComplex(0.00, 0.00);
    cuDoubleComplex matngmatmgp = make_cuDoubleComplex(0.00, 0.00);
    cuDoubleComplex matngpmatmg = make_cuDoubleComplex(0.00, 0.00);

    for(int ig=0; ig<ncouls; ++ig)
    {
        cuDoubleComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        cuDoubleComplex wtilde2 = cuDoubleComplex_square(wtilde);
        cuDoubleComplex Omega2 = cuDoubleComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
        cuDoubleComplex mygpvar1 = cuDoubleComplex_conj(aqsmtemp[n1*ncouls+igp]);
        cuDoubleComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        cuDoubleComplex matngmatmgp = cuDoubleComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
        if(ig != igp) matngpmatmg = cuDoubleComplex_product(cuDoubleComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

        double delw2, scha_mult, ssxcutoff;
        double to1 = 1e-6;
        double sexcut = 4.0;
        double gamma = 0.5;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        cuDoubleComplex sch, ssx;
    
        cuDoubleComplex wdiff = doubleMinuscuDoubleComplex(wxt , wtilde);
    
        cuDoubleComplex cden = wdiff;
        double rden = 1/cuDoubleComplex_real(cuDoubleComplex_product(cden , cuDoubleComplex_conj(cden)));
        cuDoubleComplex delw = cuDoubleComplex_mult(cuDoubleComplex_product(wtilde , cuDoubleComplex_conj(cden)) , rden);
        double delwr = cuDoubleComplex_real(cuDoubleComplex_product(delw , cuDoubleComplex_conj(delw)));
        double wdiffr = cuDoubleComplex_real(cuDoubleComplex_product(wdiff , cuDoubleComplex_conj(wdiff)));
    
        if((wdiffr > limittwo) && (delwr < limitone))
        {
            sch = cuDoubleComplex_product(delw , I_eps_array[my_igp*ngpown+ig]);
           double cden = std::pow(wxt,2);
            rden = std::pow(cden,2);
            rden = 1.00 / rden;
            ssx = cuDoubleComplex_mult(Omega2 , cden , rden);
        }
        else if (delwr > to1)
        {
            sch = expr0;
            cden = cuDoubleComplex_mult(cuDoubleComplex_product(wtilde2, doublePluscuComplex((double)0.50, delw)), 4.00);
           rden = cuDoubleComplex_real(cuDoubleComplex_product(cden , cuDoubleComplex_conj(cden)));
            rden = 1.00/rden;
            ssx = cuDoubleComplex_product(cuDoubleComplex_product(make_cuDoubleComplex(-Omega2.x, -Omega2.y) , cuDoubleComplex_conj(cden)), cuDoubleComplex_mult(delw, rden));
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }
    
        ssxcutoff = cuDoubleComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((cuDoubleComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        cuDoubleComplex_plusEquals(ssxt, cuDoubleComplex_product(matngmatmgp , ssxt));
        cuDoubleComplex_plusEquals(scht, cuDoubleComplex_product(matngmatmgp , scht));
    }
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }

    printf("********Executing Cuda + cuComplex version of the Kernel*********\n");

    auto start_totalTime = std::chrono::high_resolution_clock::now();
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);

    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int nstart = 0, nend = 3;

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];


    double to1 = 1e-6, \
    gamma = 0.5, \
    sexcut = 4.0;
    double limitone = 1.0/(to1*4.0), \
    limittwo = pow(0.5,2);

    double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value

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

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

    cuDoubleComplex expr0 = make_cuDoubleComplex(0.00, 0.00);
    cuDoubleComplex expr = make_cuDoubleComplex(0.5, 0.5);

    cuDoubleComplex *acht_n1_loc = new cuDoubleComplex[number_bands];
    cuDoubleComplex  *achtemp = new cuDoubleComplex[nend-nstart];
    cuDoubleComplex  *asxtemp = new cuDoubleComplex[nend-nstart];
    cuDoubleComplex  *aqsmtemp = new cuDoubleComplex[number_bands*ncouls];
    cuDoubleComplex  *aqsntemp = new cuDoubleComplex[number_bands*ncouls];
    cuDoubleComplex  *I_eps_array = new cuDoubleComplex[ngpown*ncouls];
    cuDoubleComplex  *wtilde_array = new cuDoubleComplex[ngpown*ncouls];
    cuDoubleComplex  *ssx_array = new cuDoubleComplex[3];
    cuDoubleComplex  *ssxa = new cuDoubleComplex[ncouls];
    cuDoubleComplex  achstemp;

    double *achtemp_re = new double[3];
    double *wx_array = new double[3];
    double *achtemp_im = new double[3];
    double *vcoul = new double[ncouls];

    printf("Executing CUDA version of the Kernel\n");
//Data Structures on Device
    cuDoubleComplex *d_wtilde_array, *d_aqsntemp, *d_aqsmtemp, *d_asxtemp;
    __shared__ cuDoubleComplex* d_I_eps_array;
    double *d_achtemp_re, *d_achtemp_im, *d_wx_array;
    double* d_vcoul;
    int *d_inv_igp_index, *d_indinv;

    if(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(cuDoubleComplex)) != cudaSuccess)
    {
        cout << "Nope could not allocate wtilde_array on device Memory needed = " << ngpown*ncouls*sizeof(cuDoubleComplex) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(cuDoubleComplex)) != cudaSuccess)
    {
        cout << "Nope could not allocate I_eps_array on device Memory needed = " << ngpown*ncouls*sizeof(cuDoubleComplex) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(cuDoubleComplex)) != cudaSuccess)
    {
        cout << "Nope could not allocate aqsntemp on device Memory needed = " << number_bands*ncouls*sizeof(cuDoubleComplex) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(cuDoubleComplex)) != cudaSuccess)
    {
        cout << "Nope could not allocate aqsmtemp on device Memory needed = " << number_bands*ncouls*sizeof(cuDoubleComplex) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_achtemp_re, 3*sizeof(double)) != cudaSuccess)
    {
        cout << "Nope could not allocate achtemp_re on device Memory needed = " << 3*sizeof(double) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_achtemp_im, 3*sizeof(double)) != cudaSuccess)
    {
        cout << "Nope could not allocate achtemp_im on device Memory needed = " << 3*sizeof(double) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_wx_array, 3*sizeof(double)) != cudaSuccess)
    {
        cout << "Nope could not allocate wx_array on device Memory needed = " << 3*sizeof(double) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)) != cudaSuccess)
    {
        cout << "Nope could not allocate vcoul on device Memory needed = " << ncouls*sizeof(double) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_indinv, (ncouls+1)*sizeof(int)) != cudaSuccess)
    {
        cout << "Nope could not allocate indinv on device Memory needed = " << (ncouls+1)*sizeof(double) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)) != cudaSuccess)
    {
        cout << "Nope could not allocate inv_igp_index on device Memory needed = " << ngpown*sizeof(double) << " bytes" << endl;
        return 0;
    }
    if(cudaMalloc((void**) &d_asxtemp, 3*sizeof(double)) != cudaSuccess)
    {
        cout << "Nope could not allocate asxtemp on device Memory needed = " << 3*sizeof(double) << " bytes" << endl;
        return 0;
    }
                        
    double occ=1.0;
    bool flag_occ;
    double achstemp_real = 0.00, achstemp_imag = 0.00;

   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = expr;
           aqsntemp[i*ncouls+j] = expr;
       }

   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = expr;
           wtilde_array[i*ncouls+j] = expr;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;


    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
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

    if(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy wtilde_array to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy I_eps_array to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy aqsntemp to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy aqsmtemp to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy vcoul to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_indinv, indinv, (ncouls+1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy indinv to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy inv_igp_index to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_wx_array, wx_array, 3*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope could not copy wx_array to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_achtemp_re, achtemp_re, 3*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Could not copy back achtemp_re to device" << endl;
        return 0;
    }
    if(cudaMemcpy(d_achtemp_im, achtemp_im, 3*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Could not copy back achtemp_re to device" << endl;
        return 0;
    }
    std::chrono::duration<double> elapsed_memcpyToDevice = std::chrono::high_resolution_clock::now() - start_withDataMovement;
    cout << "********** memcpyToDevice **********= " << elapsed_memcpyToDevice.count() << " secs" << endl;


//Start Kernel and Kernel timing
    auto start_kernelTiming = std::chrono::high_resolution_clock::now();

    till_nvbandKernel(d_aqsmtemp, d_aqsntemp, d_asxtemp, d_inv_igp_index, d_indinv, d_wtilde_array, d_wx_array, d_I_eps_array, ncouls, nvband, ngpown, nstart, nend, d_vcoul); 

    gppKernelGPU( d_wtilde_array, d_aqsntemp, d_aqsmtemp, d_I_eps_array, ncouls, ngpown, number_bands, d_wx_array, d_achtemp_re, d_achtemp_im, d_vcoul, nstart, nend, d_indinv, d_inv_igp_index);

// End Kernel and Kernel timing

//Start Synch and  timing
    cudaDeviceSynchronize();
    std::chrono::duration<double> elapsed_kernelTiming = std::chrono::high_resolution_clock::now() - start_kernelTiming;
    cout << "********** Kernel Time Taken **********= " << elapsed_kernelTiming.count() << " secs" << endl;
//End Synch and  timing

//Start memcpyToHost and  timing
    auto start_memcpyToHost = std::chrono::high_resolution_clock::now();
    if(cudaMemcpy(achtemp_re, d_achtemp_re, 3*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cout << "Could not copy back achtemp_re from device" << endl;
    
    }
    if(cudaMemcpy(achtemp_im, d_achtemp_im, 3*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cout << "Could not copy back achtemp_re from device" << endl;
        return 0;
    }

    std::chrono::duration<double> elapsed_memcpyToHost = std::chrono::high_resolution_clock::now() - start_memcpyToHost;
//    cout << "********** memcpyToHost **********= " << elapsed_memcpyToHost.count() << " secs" << endl;
//End memcpyToHost and  timing

    std::chrono::duration<double> elapsed_withDataMovement = std::chrono::high_resolution_clock::now() - start_withDataMovement;
//    cout << "********** Kernel+DataMov Time Taken **********= " << elapsed_withDataMovement.count() << " secs" << endl;
//End dataMov + Kernel and  timing

    printf(" \n Cuda Kernel Final achtemp\n");
    for(int iw=nstart; iw<nend; ++iw)
    {
        achtemp[iw] = make_cuDoubleComplex(achtemp_re[iw], achtemp_im[iw]);
//        d_print(achtemp[iw]);
        printf("( %f, %f) ", achtemp[iw].x, achtemp[iw].y);
        printf("\n");
//        achtemp[iw].print();
    }

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;

    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    cudaFree(d_wtilde_array);
    cudaFree(d_aqsntemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_I_eps_array);
    cudaFree(d_achtemp_re);
    cudaFree(d_achtemp_im);
    cudaFree(d_vcoul);
    cudaFree(d_inv_igp_index);
    cudaFree(d_indinv);
    cudaFree(d_asxtemp);

    free(acht_n1_loc);
    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(ssx_array);
    free(achtemp_re);
    free(wx_array);
    free(achtemp_im);
    free(vcoul);

    return 0;
}
