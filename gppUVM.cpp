#include "./thrustComplex.h"
//#include "./cudaAlloc.h"

#define nstart 0
#define nend 3
#define __OMPOFFLOAD__ 1
#define __reductionVersion__ 1

inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, CustomComplex  *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, CustomComplex achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads)
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

            mygpvar1 = thrust::conj(aqsmtemp[n1*ncouls+igp]);
            mygpvar2 = aqsntemp[n1*ncouls+igp];
            schs = I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = mygpvar1 * aqsntemp[n1*ncouls+igp];

            if(thrust::abs(schs) > to1)
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
        CustomComplex mygpvar1 = thrust::conj(aqsmtemp[n1*ncouls+igp]);
        CustomComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        CustomComplex matngmatmgp = aqsntemp[n1*ncouls+ig] * mygpvar1;
        if(ig != igp) matngpmatmg = thrust::conj(aqsmtemp[n1*ncouls+ig]) * mygpvar2;

        double ssxcutoff;
        double to1 = 1e-6;
        double sexcut = 4.0;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        CustomComplex sch(0.00, 0.00), ssx(0.00, 0.00);
    
        CustomComplex wdiff = wxt - wtilde;
    
        CustomComplex cden = wdiff;
        double rden = 1/(cden * thrust::conj(cden)).real();
        CustomComplex delw = wtilde * thrust::conj(cden) * rden;
        double delwr = (delw * thrust::conj(delw)).real();
        double wdiffr = (wdiff * thrust::conj(wdiff)).real();
    
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
            rden = (cden * thrust::conj(cden)).real();
            rden = 1.00/rden;
            ssx = -Omega2 * thrust::conj(cden) * delw * rden;
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }
    
        ssxcutoff = thrust::abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((thrust::abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxt += matngmatmgp * ssx;
        scht += matngmatmgp * sch;
    }
}


void till_nvband(int number_bands, int nvband, int ngpown, int ncouls, CustomComplex *asxtemp, double *wx_array, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, int *inv_igp_index, int *indinv, double *vcoul)
{
    const double occ=1.0;
#pragma omp parallel for collapse(3)
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

void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, double &elapsedKernelTimer)
{
    timeval startKernelTimer, endKernelTimer;
    //Vars to use for reduction
    double ach_re0 = 0.00, ach_re1 = 0.00, ach_re2 = 0.00, \
        ach_im0 = 0.00, ach_im1 = 0.00, ach_im2 = 0.00;

#if __OMPOFFLOAD__ 
//#pragma omp target enter data map(alloc:aqsmtemp[0:number_bands*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], \
//    aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wx_array[nstart:nend], wtilde_array[0:ngpown*ncouls])
//#pragma omp target update to(aqsmtemp[0:number_bands*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], \
    aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wx_array[nstart:nend], wtilde_array[0:ngpown*ncouls])

    gettimeofday(&startKernelTimer, NULL);

#if __reductionVersion__
#pragma omp target teams distribute parallel for collapse(2) \
    map(to:aqsmtemp[0:number_bands*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], \
    aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wx_array[nstart:nend], wtilde_array[0:ngpown*ncouls])\
    reduction(+:ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2)//\
    num_teams(number_bands*ngpown) thread_limit(32)
#else
#pragma omp target teams distribute parallel for collapse(2)\
    map(to:aqsmtemp[0:number_bands*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], \
    aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wx_array[nstart:nend], wtilde_array[0:ngpown*ncouls])\
    map(tofrom:achtemp_re[nstart:nend], achtemp_im[nstart:nend])//\
    num_teams(number_bands) //thread_limit(32)
#endif
#else
    gettimeofday(&startKernelTimer, NULL);
#pragma omp parallel for\
    reduction(+:ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2)
#endif

    for(int n1 = 0; n1<number_bands; ++n1) 
    {
//#if !__reductionVersion__
//#pragma omp parallel for
//#endif
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

//#if __reductionVersion__
//#pragma omp parallel for\
//    reduction(+:ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2)
//#endif
            for(int ig = 0; ig<ncouls; ++ig)
            {
                for(int iw = nstart; iw < nend; ++iw)
                {
//                    CustomComplex wdiff = CustomComplex_minus(&wx_array[iw], &wtilde_array[my_igp*ncouls +ig]);
//                    CustomComplex wdiff_conj = thrust::conj(&wdiff);
//                    CustomComplex delw_store1 = CustomComplex_product(&wtilde_array[my_igp*ncouls +ig], &wdiff_conj);
//                    CustomComplex delw_store2 = CustomComplex_product(&wdiff, &wdiff_conj);
//                    double delwr = 1/CustomComplex_real(&delw_store2);
//                    CustomComplex delw = CustomComplex_product(&delw_store1, &delwr);
//                    CustomComplex aqsmtemp_conj = thrust::conj(&aqsmtemp[n1*ncouls+igp]);
//                    CustomComplex sch_store1 = CustomComplex_product(&aqsmtemp_conj, &aqsntemp[n1*ncouls+igp]);
//                    CustomComplex sch_store2 = CustomComplex_product(&delw, &I_eps_array[my_igp*ncouls +ig]);
//                    CustomComplex sch_store3 = CustomComplex_product(&sch_store1, &sch_store2);
//                    CustomComplex sch_array = CustomComplex_product(&sch_store3, 0.5*vcoul[igp]);


//Using Thrust complex
                    CustomComplex wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls +ig];
                    CustomComplex wdiff_conj = thrust::conj(wdiff);
                    CustomComplex delw_store1 = wtilde_array[my_igp*ncouls +ig] * wdiff_conj;
                    CustomComplex delw_store2 = wdiff * wdiff_conj;
                    double delwr = 1/delw_store2.real();
                    CustomComplex delw = delw_store1 * delwr;
                    CustomComplex aqsmtemp_conj = thrust::conj(aqsmtemp[n1*ncouls+igp]);
                    CustomComplex sch_store1 = aqsmtemp_conj * aqsntemp[n1*ncouls+igp];
                    CustomComplex sch_store2 = delw * I_eps_array[my_igp*ncouls +ig];
                    CustomComplex sch_store3 = sch_store1 * sch_store2;
                    CustomComplex sch_array = sch_store3 * 0.5*vcoul[igp];
//                    CustomComplex sch_array = aqsmtemp[n1*ncouls + ig];
#if __reductionVersion__
                    achtemp_re_loc[iw] = sch_array.real();
                    achtemp_im_loc[iw] = sch_array.real();
#else
                    achtemp_re_loc[iw] += sch_array.real();
                    achtemp_im_loc[iw] += sch_array.imag();
#endif
                }
#if __reductionVersion__
                ach_re0 += achtemp_re_loc[0];
                ach_re1 += achtemp_re_loc[1];
                ach_re2 += achtemp_re_loc[2];
                ach_im0 += achtemp_im_loc[0];
                ach_im1 += achtemp_im_loc[1];
                ach_im2 += achtemp_im_loc[2];
#endif
            }
#if !__reductionVersion__
            for(int iw = nstart; iw < nend; ++iw)
            {
#pragma omp atomic
                achtemp_re[iw] += achtemp_re_loc[iw];
#pragma omp atomic
                achtemp_im[iw] += achtemp_im_loc[iw];
            }
#endif
        } //ngpown
    } //number_bands

    gettimeofday(&endKernelTimer, NULL);
    elapsedKernelTimer = (endKernelTimer.tv_sec - startKernelTimer.tv_sec) +1e-6*(endKernelTimer.tv_usec - startKernelTimer.tv_usec);

#if __OMPOFFLOAD__
//#pragma omp target exit data map(delete: aqsmtemp[0:number_bands*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], \
    aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wx_array[nstart:nend], wtilde_array[0:ngpown*ncouls])
#endif

#if __reductionVersion__
    achtemp_re[0] = ach_re0;
    achtemp_re[1] = ach_re1;
    achtemp_re[2] = ach_re2;
    achtemp_im[0] = ach_im0;
    achtemp_im[1] = ach_im1;
    achtemp_im[2] = ach_im2;
#endif
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }

#if __OMPOFFLOAD__
    cout << "OpenMP 4.5" << endl;
#else
    cout << "OpenMP 3.0" << endl;
#endif

//Input parameters stored in these variables.
    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int nodes_per_group = atoi(argv[4]);
    const int npes = 1; 
    const int ngpown = ncouls / (nodes_per_group * npes); 

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

    //Start the timer before the work begins.
    double elapsedKernelTimer, elapsedTimer;
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);


    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

#pragma omp target map(tofrom: numTeams, numThreads)
#pragma omp teams shared(numTeams) private(tid)
    {
        tid = omp_get_team_num();
        if(tid == 0)
        {
            numTeams = omp_get_num_teams();
#pragma omp parallel 
            {
                int ttid = omp_get_thread_num();
                if(ttid == 0)
                    numThreads = omp_get_num_threads();
            }
        }
    }
    std::cout << "Number of OpenMP Teams = " << numTeams << std::endl;
    std::cout << "Number of OpenMP DEVICE Threads = " << numThreads << std::endl;

    //Printing out the params passed.
    std::cout << "Sizeof(CustomComplex = " << sizeof(CustomComplex) << " bytes" << std::endl;
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart << endl;
   
    CustomComplex expr0(0.00, 0.00);
    CustomComplex expr(0.5, 0.5);
    long double memFootPrint = 0.00;

    //ALLOCATE statements from fortran gppkernel.
    CustomComplex *acht_n1_loc = new CustomComplex[number_bands];
    memFootPrint += number_bands*sizeof(CustomComplex);

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

    //0-nvband iterations
    till_nvband(number_bands, nvband, ngpown, ncouls, asxtemp, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, inv_igp_index, indinv, vcoul);

    //reduction on achstemp
#pragma omp parallel for 
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

    noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, elapsedKernelTimer);

    gettimeofday(&endTimer, NULL);
    elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    printf("\n Final achtemp\n");
    for(int iw=nstart; iw<nend; ++iw)
    {
        CustomComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
    }
    cout << "achtemp[0] = (" << achtemp[0].real() << "," << achtemp[0].imag() << ")" << endl;
//        achtemp[0].print();

    cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsedTimer << " secs" << endl;

    free(acht_n1_loc);
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
