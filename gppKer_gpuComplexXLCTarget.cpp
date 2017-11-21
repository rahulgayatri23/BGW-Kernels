#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

#include "GPUComplex.h"
//#include "Complex_target.cpp"
#define igblk 512

using namespace std;
int debug = 0;

#pragma omp declare target
inline void flagOCC_solver(double wxt, GPUComplex *wtilde_array, int my_igp, int n1, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex &ssxt, GPUComplex &scht,int ncouls, int igp, int number_bands, int ngpown);
inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, GPUComplex  *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads);
#pragma omp end declare target

//#define CACHE_LINE 32
//#define CACHE_ALIGN __declspec(align(CACHE_LINE))
inline double local_abs(std::complex<double> compl_num)
{
    double re = real(compl_num) * real(compl_num);
    double im = imag(compl_num) * imag(compl_num);

    double result = sqrt(re+im);
    return result;
}

std::complex<double> local_pow(std::complex<double> compl_num, int n)
{
    double re = std::real(compl_num);
    double im = std::imag(compl_num);

    std::complex<double> result(re*re - im*im, 2*re*im);
    return result;
}


void ssxt_scht_solver(double wxt, int igp, int my_igp, int ig, std::complex<double> wtilde, std::complex<double> wtilde2, std::complex<double> Omega2, std::complex<double> matngmatmgp, std::complex<double> matngpmatmg, std::complex<double> mygpvar1, std::complex<double> mygpvar2, std::complex<double>& ssxa, std::complex<double>& scha, std::complex<double> I_eps_array_igp_myIgp)
{
    std::complex<double> expr0( 0.0 , 0.0);
    double delw2, scha_mult, ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    std::complex<double> sch, ssx;

    std::complex<double> wdiff = wxt - wtilde;

    std::complex<double> cden = wdiff;
    double rden = 1/real(cden * conj(cden));
    std::complex<double> delw = wtilde * conj(cden) * rden;
    double delwr = real(delw * conj(delw));
    double wdiffr = real(wdiff * conj(wdiff));

    if((wdiffr > limittwo) && (delwr < limitone))
    {
        sch = delw * I_eps_array_igp_myIgp;
        cden = pow(wxt,2);
        rden = real(cden * conj(cden));
        rden = 1.00 / rden;
        ssx = Omega2 * conj(cden) * rden;
    }
    else if (delwr > to1)
    {
        sch = expr0;
        cden = (double) 4.00 * wtilde2 * (delw + (double)0.50);
        rden = real(cden * conj(cden));
        rden = 1.00/rden;
        ssx = -Omega2 * conj(cden) * rden * delw;
    }
    else
    {
        sch = expr0;
        ssx = expr0;
    }

    ssxcutoff = sexcut*local_abs(I_eps_array_igp_myIgp);
    if((local_abs(ssx) > ssxcutoff) && (local_abs(wxt) < 0.00)) ssx = 0.00;

    ssxa = matngmatmgp*ssx;
    scha = matngmatmgp*sch;
}

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
            matngmatmgp = GPUComplex_product(mygpvar1, aqsntemp[n1*ncouls+igp]);


            if(GPUComplex_abs(schs) > to1)
                GPUComplex_fma(schstemp, matngmatmgp, schs);
            }
            else 
            {
                for(int ig=1; ig<ncouls; ++ig)
                {
                    GPUComplex mult_result(GPUComplex_product(I_eps_array[my_igp*ncouls+ig] , mygpvar1));
                    GPUComplex_fms(schstemp,aqsntemp[n1*ncouls+igp], mult_result); 
                }
            }

        schstemp = GPUComplex_mult(schstemp, vcoul[igp], 0.5);
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
        GPUComplex wtilde2 = GPUComplex_square(wtilde);
        GPUComplex Omega2 = GPUComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
        GPUComplex mygpvar1 = GPUComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPUComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        GPUComplex matngmatmgp = GPUComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
        if(ig != igp) matngpmatmg = GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

        double delw2, scha_mult, ssxcutoff;
        double to1 = 1e-6;
        double sexcut = 4.0;
        double gamma = 0.5;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        GPUComplex sch, ssx;
    
        GPUComplex wdiff = doubleMinusGPUComplex(wxt , wtilde);
    
        GPUComplex cden = wdiff;
        double rden = 1/GPUComplex_real(GPUComplex_product(cden , GPUComplex_conj(cden)));
        GPUComplex delw = GPUComplex_mult(GPUComplex_product(wtilde , GPUComplex_conj(cden)) , rden);
        double delwr = GPUComplex_real(GPUComplex_product(delw , GPUComplex_conj(delw)));
        double wdiffr = GPUComplex_real(GPUComplex_product(wdiff , GPUComplex_conj(wdiff)));
    
        if((wdiffr > limittwo) && (delwr < limitone))
        {
            sch = GPUComplex_product(delw , I_eps_array[my_igp*ngpown+ig]);
            double cden = std::pow(wxt,2);
            rden = std::pow(cden,2);
            rden = 1.00 / rden;
            ssx = GPUComplex_mult(Omega2 , cden , rden);
        }
        else if (delwr > to1)
        {
            sch = expr0;
            cden = GPUComplex_mult(GPUComplex_product(wtilde2, doublePlusGPUComplex((double)0.50, delw)), 4.00);
            rden = GPUComplex_real(GPUComplex_product(cden , GPUComplex_conj(cden)));
            rden = 1.00/rden;
            ssx = GPUComplex_product(GPUComplex_product(-Omega2 , GPUComplex_conj(cden)), GPUComplex_mult(delw, rden));
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }
    
        ssxcutoff = GPUComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((GPUComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxt += GPUComplex_product(matngmatmgp , ssx);
        scht += GPUComplex_product(matngmatmgp , sch);
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

    int* inv_igp_index = new int[ngpown];
    int* indinv = new int[ncouls+1];

    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

#pragma omp target enter data map(alloc: numTeams, numThreads)
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
#pragma omp target exit data map(delete: numTeams, numThreads)
    std::cout << "Number of OpenMP Teams = " << numTeams << std::endl;
    std::cout << "Number of OpenMP DEVICE Threads = " << numThreads << std::endl;

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


    //ALLOCATE statements from fortran gppkernel.
    
   
    GPUComplex expr0(0.00, 0.00);
    GPUComplex expr(0.5, 0.5);

    GPUComplex *acht_n1_loc = new GPUComplex[number_bands];
    GPUComplex *achtemp = new GPUComplex[nend-nstart];
    GPUComplex *asxtemp = new GPUComplex[nend-nstart];
    GPUComplex *aqsmtemp = new GPUComplex[number_bands*ncouls];
    GPUComplex *aqsntemp = new GPUComplex[number_bands*ncouls];
    GPUComplex *I_eps_array = new GPUComplex[ngpown*ncouls];
    GPUComplex *wtilde_array = new GPUComplex[ngpown*ncouls];
    GPUComplex *ssx_array = new GPUComplex[3];
    GPUComplex *ssxa = new GPUComplex[ncouls];
    GPUComplex *scha = new GPUComplex[igblk];
    GPUComplex achstemp;

    double *achtemp_re = new double[3];
    double *achtemp_im = new double[3];
                        
    double *vcoul = new double[ncouls];
    double wx_array[3];
    double occ=1.0;
    bool flag_occ;
    double achstemp_real = 0.00, achstemp_imag = 0.00;
    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;


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
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls+1;

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

    auto start_chrono = std::chrono::high_resolution_clock::now();

#pragma omp target teams distribute parallel for map(to:aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1])\
    map(tofrom:achstemp) nowait
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

#pragma omp target enter data map(alloc:asxtemp[nstart:nend], acht_n1_loc[0:number_bands], aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], achtemp_re[nstart:nend], achtemp_im[nstart:nend])

#pragma omp target update to(aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], wtilde_array[0:ngpown*ncouls], asxtemp[nstart:nend], achtemp_re[nstart:nend], achtemp[nstart:nend])

#pragma omp target teams distribute parallel for collapse(3) map(to:aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1])\
    map(tofrom:asxtemp[nstart:nend]) nowait
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
                    
                    asxtemp[iw] += GPUComplex_mult(ssxt , occ , vcoul[igp]);
              }
            }

       }

#pragma omp target teams distribute parallel for shared(vcoul, aqsntemp, aqsmtemp, I_eps_array) collapse(2) map(to:aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1])\
    map(tofrom:acht_n1_loc[0:number_bands], asxtemp[nstart:nend], achtemp_re[nstart:nend], achtemp[nstart:nend], wx_array[nstart:nend]) 
    for(int n1 = 0; n1<number_bands; ++n1) 
    {
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            GPUComplex sch_array[3];
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if(indigp == ncouls)
                igp = ncouls-1;

            GPUComplex mygpvar1 = GPUComplex_conj(aqsmtemp[n1*ncouls+igp]);
            GPUComplex wdiff, delw;
            double achtemp_re_loc[3], \
                achtemp_im_loc[3];

/*iw-inside VERSION*/
            for(int iw=nstart; iw<nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

                for(int ig = 0; ig<ncouls; ++ig)
                {
                    int iw = nstart;
                    achtemp_re_loc[iw] += GPUComplex_real(GPUComplex_mult(GPUComplex_mult(GPUComplex_product(GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), GPUComplex_product(GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))), 1/GPUComplex_real(GPUComplex_product(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]), GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))))), I_eps_array[my_igp*ncouls+ig])), 0.5), vcoul[igp]));

                    achtemp_im_loc[iw] += GPUComplex_imag(GPUComplex_mult(GPUComplex_mult(GPUComplex_product(GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), GPUComplex_product(GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))), 1/GPUComplex_real(GPUComplex_product(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]), GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))))), I_eps_array[my_igp*ncouls+ig])), 0.5), vcoul[igp]));

                    iw++;
                    achtemp_re_loc[iw] += GPUComplex_real(GPUComplex_mult(GPUComplex_mult(GPUComplex_product(GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), GPUComplex_product(GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))), 1/GPUComplex_real(GPUComplex_product(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]), GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))))), I_eps_array[my_igp*ncouls+ig])), 0.5), vcoul[igp]));
                    achtemp_im_loc[iw] += GPUComplex_imag(GPUComplex_mult(GPUComplex_mult(GPUComplex_product(GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), GPUComplex_product(GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))), 1/GPUComplex_real(GPUComplex_product(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]), GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))))), I_eps_array[my_igp*ncouls+ig])), 0.5), vcoul[igp]));

                    iw++;
                    achtemp_re_loc[iw] += GPUComplex_real(GPUComplex_mult(GPUComplex_mult(GPUComplex_product(GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), GPUComplex_product(GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))), 1/GPUComplex_real(GPUComplex_product(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]), GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))))), I_eps_array[my_igp*ncouls+ig])), 0.5), vcoul[igp]));
                    achtemp_im_loc[iw] += GPUComplex_imag(GPUComplex_mult(GPUComplex_mult(GPUComplex_product(GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), GPUComplex_product(GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))), 1/GPUComplex_real(GPUComplex_product(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]), GPUComplex_conj(doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]))))), I_eps_array[my_igp*ncouls+ig])), 0.5), vcoul[igp]));

                }

            for(int iw=nstart; iw<nend; ++iw)
            {
#pragma omp atomic
                achtemp_re[iw] += achtemp_re_loc[iw];
#pragma omp atomic
                achtemp_im[iw] += achtemp_im_loc[iw];
            }
            
            acht_n1_loc[n1] += GPUComplex_mult(sch_array[2] , vcoul[igp]);
        } //ngpown
    } // number-bands

    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;

#pragma omp target update from (acht_n1_loc[0:number_bands], asxtemp[0:(nend-nstart)], achtemp_re[nstart:nend], achtemp_im[nstart:nend] )

#pragma omp target exit data map(delete: acht_n1_loc[:0], aqsmtemp[:0],aqsntemp[:0], I_eps_array[:0], wtilde_array[:0], vcoul[:0], inv_igp_index[:0], indinv[:0], asxtemp[:0], scha[:0])

    printf(" \n Final achstemp\n");
    achstemp.print();

    printf("\n Final achtemp\n");
    for(int iw=nstart; iw<nend; ++iw)
    {
        GPUComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;
    cout << "********** Kernel Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    free(acht_n1_loc);
    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(vcoul);
    free(ssx_array);

    return 0;
}
