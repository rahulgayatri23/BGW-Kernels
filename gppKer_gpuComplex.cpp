#include "Complex.h"

using namespace std;
#define nstart 0
#define nend 3

inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, GPUComplex  *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads)
{
    double to1 = 1e-6;
    GPUComplex schstemp(0.0, 0.0);;

    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        GPUComplex schs(0.0, 0.0);
        GPUComplex matngmatmgp(0.0, 0.0);
        GPUComplex matngpmatmg(0.0, 0.0);
        GPUComplex halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

            GPUComplex mygpvar2, mygpvar1;
            mygpvar1 = GPUComplex_conj(aqsmtemp[n1*ncouls+igp]);
            mygpvar2 = aqsntemp[n1*ncouls+igp];



            schs = I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = mygpvar1 * aqsntemp[n1*ncouls+igp];


            if(GPUComplex_abs(schs) > to1)
                schstemp += matngmatmgp * schs;
            }
            else 
            {
                for(int ig=1; ig<ncouls; ++ig)
                {
                    GPUComplex mult_result(I_eps_array[my_igp*ncouls+ig] * mygpvar1);
                    schstemp -= aqsntemp[n1*ncouls +igp] * mult_result;
                }
            }

        schstemp = schstemp * vcoul[igp]*0.5;
        achstemp += schstemp;
    }
}

inline void flagOCC_solver(double wxt, GPUComplex *wtilde_array, int my_igp, int n1, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex &ssxt, GPUComplex &scht,int ncouls, int igp, int number_bands, int ngpown)
{
    GPUComplex expr0(0.00, 0.00);
    GPUComplex expr(0.5, 0.5);
    GPUComplex matngmatmgp(0.0, 0.0);
    GPUComplex matngpmatmg(0.0, 0.0);

    for(int ig=0; ig<ncouls; ++ig)
    {
        GPUComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        GPUComplex wtilde2 = wtilde * wtilde;
        GPUComplex Omega2 = wtilde2*I_eps_array[my_igp*ncouls+ig];
        GPUComplex mygpvar1 = GPUComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPUComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        GPUComplex matngmatmgp = aqsntemp[n1*ncouls+ig] * mygpvar1;
        if(ig != igp) matngpmatmg = GPUComplex_conj(aqsmtemp[n1*ncouls+ig]) * mygpvar2;

        double delw2, scha_mult, ssxcutoff;
        double to1 = 1e-6;
        double sexcut = 4.0;
        double gamma = 0.5;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        GPUComplex sch, ssx;
    
        GPUComplex wdiff = wxt - wtilde;
    
        GPUComplex cden = wdiff;
        double rden = 1/GPUComplex_real(cden * GPUComplex_conj(cden));
        GPUComplex delw = wtilde * GPUComplex_conj(cden) * rden;
        double delwr = GPUComplex_real(delw * GPUComplex_conj(delw));
        double wdiffr = GPUComplex_real(wdiff * GPUComplex_conj(wdiff));
    
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
            rden = GPUComplex_real(cden * GPUComplex_conj(cden));
            rden = 1.00/rden;
            ssx = -Omega2 * GPUComplex_conj(cden) * delw * rden;
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }
    
        ssxcutoff = GPUComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((GPUComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxt += matngmatmgp * ssx;
        scht += matngmatmgp * sch;
    }
}


void till_nvband(int number_bands, int nvband, int ngpown, int ncouls, GPUComplex *asxtemp, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, int *inv_igp_index, int *indinv, double *vcoul)
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
                 GPUComplex ssxt(0.00, 0.00);
                 GPUComplex scht(0.00, 0.00);
                 flagOCC_solver(wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, number_bands, ngpown);
                 asxtemp[iw] += ssxt * occ * vcoul[igp];
           }
         }
    }
}

static inline void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im)
{
    for(int n1 = 0; n1<number_bands; ++n1) 
    {
#pragma omp parallel for  default(shared) firstprivate(ngpown, ncouls, number_bands)
    for(int my_igp=0; my_igp<ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];

        GPUComplex wdiff, delw;

        double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
        for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

        for(int ig = 0; ig<ncouls; ++ig)
        {
            for(int iw = nstart; iw < nend; ++iw)
            {
                wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                delw = wtilde_array[my_igp*ncouls+ig] * GPUComplex_conj(wdiff) * (1/GPUComplex_real((wdiff * GPUComplex_conj(wdiff)))); 
                GPUComplex sch_array = (((GPUComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig]) * (delw * I_eps_array[my_igp*ncouls+ig])) * 0.5*vcoul[igp]);
                achtemp_re_loc[iw] += GPUComplex_real(sch_array);
                achtemp_im_loc[iw] += GPUComplex_imag(sch_array);
            }
        }
        for(int iw = nstart; iw < nend; ++iw)
        {
#pragma omp atomic
            achtemp_re[iw] += achtemp_re_loc[iw];
#pragma omp atomic
            achtemp_im[iw] += achtemp_im_loc[iw];
        }
    } //ngpown
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


    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

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
   
    GPUComplex expr0(0.00, 0.00);
    GPUComplex expr(0.5, 0.5);
    long double memFootPrint = 0.00;

    //ALLOCATE statements from fortran gppkernel.
    GPUComplex *acht_n1_loc = new GPUComplex[number_bands];
    memFootPrint += number_bands*sizeof(GPUComplex);

    GPUComplex *achtemp = new GPUComplex[nend-nstart];
    GPUComplex *asxtemp = new GPUComplex[nend-nstart];
    GPUComplex *ssx_array = new GPUComplex[nend-nstart];
    memFootPrint += 3*(nend-nstart)*sizeof(GPUComplex);

    GPUComplex *aqsmtemp = new GPUComplex[number_bands*ncouls];
    GPUComplex *aqsntemp = new GPUComplex[number_bands*ncouls];
    memFootPrint += 2*(number_bands*ncouls)*sizeof(GPUComplex);

    GPUComplex *I_eps_array = new GPUComplex[ngpown*ncouls];
    GPUComplex *wtilde_array = new GPUComplex[ngpown*ncouls];
    memFootPrint += 2*(ngpown*ncouls)*sizeof(GPUComplex);

    GPUComplex *ssxa = new GPUComplex[ncouls];
    double *vcoul = new double[ncouls];
    memFootPrint += ncouls*sizeof(GPUComplex);
    memFootPrint += ncouls*sizeof(double);

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];
    memFootPrint += ngpown*sizeof(int);
    memFootPrint += (ncouls+1)*sizeof(int);

//Real and imaginary parts of achtemp calculated separately to avoid critical.
    double *achtemp_re = new double[nend-nstart];
    double *achtemp_im = new double[nend-nstart];
    memFootPrint += 2*sizeof(double);

    double wx_array[nend-nstart];
    GPUComplex achstemp;
                        
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


    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0, tmp=1; ig<ncouls; ++ig,tmp++)
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

    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

    //0-nvband iterations
    till_nvband(number_bands, nvband, ngpown, ncouls, asxtemp, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, inv_igp_index, indinv, vcoul);

    //reduction on achstemp
#pragma omp parallel for 
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

    //main-loop with output on achtemp divide among achtemp_re && achtemp_im
//    noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im);
    for(int n1 = 0; n1<number_bands; ++n1) 
    {
#pragma omp parallel for  collapse(2) default(shared) firstprivate(ngpown, ncouls, number_bands)
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            GPUComplex wdiff, delw;

            double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

            for(int ig = 0; ig<ncouls; ++ig)
            {
                for(int iw = nstart; iw < nend; ++iw)
                {
                    wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                    delw = wtilde_array[my_igp*ncouls+ig] * GPUComplex_conj(wdiff) * (1/GPUComplex_real((wdiff * GPUComplex_conj(wdiff)))); 
                    GPUComplex sch_array = (((GPUComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig]) * (delw * I_eps_array[my_igp*ncouls+ig])) * 0.5*vcoul[igp]);
                    achtemp_re_loc[iw] += GPUComplex_real(sch_array);
                    achtemp_im_loc[iw] += GPUComplex_imag(sch_array);
                }
            }
            for(int iw = nstart; iw < nend; ++iw)
            {
#pragma omp atomic
                achtemp_re[iw] += achtemp_re_loc[iw];
#pragma omp atomic
                achtemp_im[iw] += achtemp_im_loc[iw];
            }
        } //ngpown
    }

    //Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    printf(" \n Final achstemp\n");
    achstemp.print();

    printf("\n Final achtemp\n");

    for(int iw=nstart; iw<nend; ++iw)
    {
        GPUComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    cout << "********** Kernel Time Taken **********= " << elapsedTimer << " secs" << endl;

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
