#include "CustomComplex.h"

#define CUDAVER 1

/*Contains stride dimenion*/

using namespace std;

inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, CustomComplex  *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, CustomComplex achstemp,  int* indinv, int ngpown, double* vcoul)
{
    double to1 = 1e-6;
    CustomComplex schstemp(0.0, 0.0);;

    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        CustomComplex schs(0.0, 0.0);
        CustomComplex matngmatmgp(0.0, 0.0);
        CustomComplex matngpmatmg(0.0, 0.0);
        CustomComplex mygpvar1(0.00, 0.00), mygpvar2(0.00, 0.00);
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

            mygpvar1 = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]);
            mygpvar2 = aqsntemp[n1*ncouls+igp];
            schs = I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = mygpvar1 * aqsntemp[n1*ncouls+igp];

            if(CustomComplex_abs(schs) > to1)
                schstemp += matngmatmgp * schs;
            }
            else 
            {
                for(int ig=1; ig<ncouls; ++ig)
                {
                    CustomComplex mult_result(I_eps_array[my_igp*ncouls+ig] * mygpvar1);
                    schstemp -= aqsntemp[n1*ncouls +igp] * mult_result;
                }
            }

        schstemp = schstemp * vcoul[igp]*0.5;
        achstemp += schstemp;
    }
}

inline void flagOCC_solver(double wxt, CustomComplex *wtilde_array, int my_igp, int n1, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, CustomComplex &ssxt, CustomComplex &scht,int ncouls, int igp, int number_bands, int ngpown)
{
    CustomComplex expr0(0.00, 0.00);
    CustomComplex expr(0.5, 0.5);
    CustomComplex matngmatmgp(0.0, 0.0);
    CustomComplex matngpmatmg(0.0, 0.0);

    for(int ig=0; ig<ncouls; ++ig)
    {
        CustomComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        CustomComplex wtilde2 = wtilde * wtilde;
        CustomComplex Omega2 = wtilde2*I_eps_array[my_igp*ncouls+ig];
        CustomComplex mygpvar1 = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]);
        CustomComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        CustomComplex matngmatmgp = aqsntemp[n1*ncouls+ig] * mygpvar1;
        if(ig != igp) matngpmatmg = CustomComplex_conj(aqsmtemp[n1*ncouls+ig]) * mygpvar2;

        double ssxcutoff;
        double to1 = 1e-6;
        double sexcut = 4.0;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        CustomComplex sch(0.00, 0.00), ssx(0.00, 0.00);
    
        CustomComplex wdiff = wxt - wtilde;
    
        CustomComplex cden = wdiff;
        double rden = 1/CustomComplex_real(cden * CustomComplex_conj(cden));
        CustomComplex delw = wtilde * CustomComplex_conj(cden) * rden;
        double delwr = CustomComplex_real(delw * CustomComplex_conj(delw));
        double wdiffr = CustomComplex_real(wdiff * CustomComplex_conj(wdiff));
    
        if((wdiffr > limittwo) && (delwr < limitone))
        {
            sch = delw * I_eps_array[my_igp*ngpown+ig];
            double cden = std::pow(wxt,2);
            rden = std::pow(cden,2);
            rden = 1.00 / rden;
            ssx = Omega2 * cden * rden;
        }
        else if (delwr > to1)
        {
            sch = expr0;
            cden = wtilde2 * (0.50 + delw) * 4.00;
            rden = CustomComplex_real(cden * CustomComplex_conj(cden));
            rden = 1.00/rden;
            ssx = -Omega2 * CustomComplex_conj(cden) * delw * rden;
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }
    
        ssxcutoff = CustomComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((CustomComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxt += matngmatmgp * ssx;
        scht += matngmatmgp * sch;
    }
}


void till_nvband(int number_bands, int nvband, int ngpown, int ncouls, CustomComplex *asxtemp, double *wx_array, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, int *inv_igp_index, int *indinv, double *vcoul)
{
    const double occ=1.0;
    for(int n1 = 0; n1 < nvband; n1++)
    {
         for(int my_igp=0; my_igp<ngpown; ++my_igp)
         {
            for(int iw=nstart; iw<nend; iw++)
            {
                 int indigp = inv_igp_index[my_igp];
                 int igp = indinv[indigp];
                 CustomComplex ssxt(0.00, 0.00);
                 CustomComplex scht(0.00, 0.00);
                 flagOCC_solver(wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, number_bands, ngpown);
                 asxtemp[iw] += ssxt * occ * vcoul[igp];
           }
         }
    }
}

void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, int stride)
{
    if(stride == 0)
    {
        for(int n1 = 0; n1<number_bands; ++n1) 
        {
            for(int my_igp=0; my_igp<ngpown; ++my_igp)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];

                CustomComplex wdiff(0.00, 0.00), delw(0.00, 0.00);

                double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
                for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

                for(int ig = 0; ig<ncouls; ++ig)
                {
                    for(int iw = nstart; iw < nend; ++iw)
                    {
                        wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig]; //2 flops

                        //2 conj + 2 * product + 1 mult + 1 divide = 17
                        delw = wtilde_array[my_igp*ncouls+ig] * CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff)))); 

                        //1 conj + 3 product + 1 mult + 1 real-mult = 22
//                        CustomComplex sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];
                        CustomComplex sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+ig]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[ig];

                        //2 flops
                        achtemp_re_loc[iw] += CustomComplex_real(sch_array);
                        achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
                    }
                }
                for(int iw = nstart; iw < nend; ++iw)
                {
                    achtemp_re[iw] += achtemp_re_loc[iw];
                    achtemp_im[iw] += achtemp_im_loc[iw];
                }
            } //ngpown
        } //number_bands
    } //if-loop


    else
    {
        for(int n1 = 0; n1<number_bands; ++n1) 
        {
            for(int my_igp=0; my_igp<ngpown; ++my_igp)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];

                CustomComplex wdiff(0.00, 0.00), delw(0.00, 0.00);

                double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
                for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

                for(int igmin = 0; igmin<stride; igmin++)
                {
                    for(int ig=igmin; ig<ncouls; ig+=stride)
                    {
                        for(int iw = nstart; iw < nend; ++iw)
                        {
                            wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig]; //2 flops

                            //2 conj + 2 * product + 1 mult + 1 divide = 17
                            delw = wtilde_array[my_igp*ncouls+ig] * CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff)))); 

                            //1 conj + 3 product + 1 mult + 1 real-mult = 22
                            CustomComplex sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];

                            //2 flops
                            achtemp_re_loc[iw] += CustomComplex_real(sch_array);
                            achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
                        }
                    }
                }
                for(int iw = nstart; iw < nend; ++iw)
                {
                    achtemp_re[iw] += achtemp_re_loc[iw];
                    achtemp_im[iw] += achtemp_im_loc[iw];
                }
            } //ngpown
        } //number_bands
    } //else
}

int main(int argc, char** argv)
{

    if (argc != 6)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> <stride> " << endl;
        exit (0);
    }

//Input parameters stored in these variables.
    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int nodes_per_group = atoi(argv[4]);
    const int npes = 1; 
    const int ngpown = ncouls / (nodes_per_group * npes); 
    const int stride = atoi(argv[5]);

//Constants that will be used later
    const double e_lk = 10;
    const double dw = 1;
    const double to1 = 1e-6;
    const double gamma = 0.5;
    const double sexcut = 4.0;
    const double limitone = 1.0/(to1*4.0);
    const double limittwo = pow(0.5,2);
    const double e_n1kq= 6.0; 
    const double occ=1.0;

    //Printing out the params passed.
#if CUDAVER
    std::cout << "Cuda version " << std::endl;
#else 
    std::cout << "Seq version " << std::endl;
#endif
    
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t stride = " << stride << endl;
   
    CustomComplex expr0(0.00, 0.00);
    CustomComplex expr(0.5, 0.5);
    long double memFootPrint = 0.00;

    //ALLOCATE statements from fortran gppkernel.
    CustomComplex *achtemp = new CustomComplex[nend-nstart];
    CustomComplex *asxtemp = new CustomComplex[nend-nstart];
    CustomComplex *ssx_array = new CustomComplex[nend-nstart];
    memFootPrint += 3*(nend-nstart)*sizeof(CustomComplex);

    CustomComplex *aqsmtemp = new CustomComplex[number_bands*ncouls];
    CustomComplex *aqsntemp = new CustomComplex[number_bands*ncouls];
    memFootPrint += 2*(number_bands*ncouls)*sizeof(CustomComplex);

    CustomComplex *I_eps_array = new CustomComplex[ngpown*ncouls];
    CustomComplex *wtilde_array = new CustomComplex[ngpown*ncouls];
    memFootPrint += 2*(ngpown*ncouls)*sizeof(CustomComplex);

    CustomComplex *ssxa = new CustomComplex[ncouls];
    double *vcoul = new double[ncouls];
    memFootPrint += ncouls*sizeof(CustomComplex);
    memFootPrint += ncouls*sizeof(double);

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];
    memFootPrint += ngpown*sizeof(int);
    memFootPrint += (ncouls+1)*sizeof(int);

//Real and imaginary parts of achtemp calculated separately to avoid critical.
    double *achtemp_re = new double[nend-nstart];
    double *achtemp_im = new double[nend-nstart];
    memFootPrint += 2*(nend-nstart)*sizeof(double);

    double wx_array[nend-nstart];
    CustomComplex achstemp;
                        
    //Print Memory Foot print 
    cout << "Memory Foot Print = " << memFootPrint / pow(1024,3) << " GBs" << endl;

#if CUDAVER
    //Cuda related declare && allocate
    //Memory pointer declarations for device
    CustomComplex *d_asxtemp, *d_aqsmtemp, *d_aqsntemp, *d_I_eps_array, *d_wtilde_array;
    double *d_achtemp_re, *d_achtemp_im, *d_wx_array, *d_vcoul;
    int *d_inv_igp_index, *d_indinv;

    //Allocate memory on the device
    CudaSafeCall(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_asxtemp, (nend-nstart)*sizeof(CustomComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_re, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_im, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_wx_array, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_indinv, (ncouls+1)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)));
#endif

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


    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

       for(int iw=nstart; iw<nend; ++iw)
       {
           asxtemp[iw] = expr0;
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
       }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

#if CUDAVER
    //Memcpy to device

    CudaSafeCall(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_asxtemp, asxtemp, (nend-nstart)*sizeof(CustomComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_re, achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_im, achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wx_array, wx_array, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_inv_igp_index, vcoul, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_indinv, indinv, (ncouls+1)*sizeof(int), cudaMemcpyHostToDevice));
#endif


    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

    //0-nvband iterations
    till_nvband(number_bands, nvband, ngpown, ncouls, asxtemp, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, inv_igp_index, indinv, vcoul);

    //reduction on achstemp
#pragma omp parallel for 
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul);

    //main-loop with output on achtemp divide among achtemp_re && achtemp_im
#if !CUDAVER
        noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, stride);
#else
        d_noflagOCC_solver(number_bands, ngpown, ncouls, d_inv_igp_index, d_indinv, d_wx_array, d_wtilde_array, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_vcoul, d_achtemp_re, d_achtemp_im, stride);

    //Copy the results back
    CudaSafeCall(cudaMemcpy(achtemp_re, d_achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achtemp_im, d_achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));
#endif

    //Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    printf(" \n Final achstemp\n");
    achstemp.print();

    printf("\n Final achtemp\n");

    for(int iw=nstart; iw<nend; ++iw)
    {
        CustomComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
//        achtemp[iw].print();
    }
        achtemp[0].print();

    cout << "********** Kernel Time Taken **********= " << elapsedTimer << " secs" << endl;

    //CudaFree
#if CUDAVER
    cudaFree(d_asxtemp);
    cudaFree(d_aqsmtemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_I_eps_array);
    cudaFree(d_wtilde_array);
    cudaFree(d_achtemp_re);
    cudaFree(d_achtemp_im);
    cudaFree(d_wx_array);
    cudaFree(d_vcoul);
    cudaFree(d_indinv);
    cudaFree(d_inv_igp_index);
#endif

    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(vcoul);
    free(ssx_array);
    free(inv_igp_index);
    free(indinv);

    return 0;
}
