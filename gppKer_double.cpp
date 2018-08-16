#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <sys/time.h>
#include <ittnotify.h>

using namespace std;
#define nstart 0
#define nend 6

inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, double  *aqsmtemp, double *aqsntemp, double *I_eps_array, double achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads)
{
    double to1 = 1e-6;
    double schstemp = 0.0;

    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        double schs = 0.00;
        double matngmatmgp = 0.0;
        double matngpmatmg = 0.0;
        double halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

            double mygpvar2, mygpvar1;
            mygpvar1 = aqsmtemp[n1*ncouls+igp];
            mygpvar2 = aqsntemp[n1*ncouls+igp];

            schs = I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = mygpvar1 * aqsntemp[n1*ncouls+igp];

            if(schs > to1)
                schstemp += matngmatmgp * schs;
            }
            else 
            {
                for(int ig=1; ig<ncouls; ++ig)
                    schstemp -= aqsntemp[n1*ncouls + igp] * (I_eps_array[my_igp*ncouls + ig] * mygpvar1);
            }

        schstemp *= vcoul[igp] * 0.5;
        achstemp += schstemp;
    }
}

inline void flagOCC_solver(int *inv_igp_index, int *indinv, double *wx_array, double *wtilde_array, double *aqsmtemp, double *aqsntemp, double *I_eps_array,int nvband, int ncouls, int number_bands, int ngpown, double *asxtemp)
{
#pragma omp parallel for collapse(3)
    for(int n1 = 0; n1 < nvband; n1++)
    {
        for(int my_igp = 0; my_igp<ngpown; ++my_igp)
        {
           for(int iw=nstart; iw<nend; iw++)
           {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                double ssxt = 0.00;
                double scht = 0.00;
                double matngmatmgp = 0.00;
                double matngpmatmg = 0.00;
                for(int ig=0; ig<ncouls; ++ig)
                {
                    double wtilde = wtilde_array[my_igp*ncouls+ig];
                    double wtilde2 = wtilde * wtilde;
                    double Omega2 = wtilde2*I_eps_array[my_igp*ncouls+ig];
                    double mygpvar1 = aqsmtemp[n1*ncouls+igp];
                    double mygpvar2 = aqsmtemp[n1*ncouls+igp];
                    double matngmatmgp = aqsntemp[n1*ncouls+ig] * mygpvar1;
                    if(ig != igp) matngpmatmg = aqsmtemp[n1*ncouls+ig] * mygpvar2;

                    double ssxcutoff;
                    double to1 = 1e-6;
                    double sexcut = 4.0;
                    double gamma = 0.5;
                    double limitone = 1.0/(to1*4.0);
                    double limittwo = pow(0.5,2);
                    double sch, ssx;
                
                    double wdiff = wx_array[iw] - wtilde;
                
                    double cden = wdiff;
                    double rden = 1/cden * cden;
                    double delw = wtilde * cden * rden;
                    double delwr = delw * delw;
                    double wdiffr = wdiff * wdiff;
                
                    if((wdiffr > limittwo) && (delwr < limitone))
                    {
                        sch = delw * I_eps_array[my_igp*ngpown+ig];
                        double cden = wx_array[iw] * wx_array[iw];
                        rden = cden * cden;
                        rden = 1.00 / rden;
                        ssx = Omega2 * cden * rden;
                    }
                    else if (delwr > to1)
                    {
                        sch = 0.00;
                        cden = wtilde2*0.50*delw*4.00;
                        rden = cden * cden;
                        rden = 1.00/rden;
                        ssx = -Omega2 * cden * delw * rden;
                    }
                    else
                    {
                        sch = 0.00;
                        ssx = 0.00;
                    }
                
                    ssxcutoff = I_eps_array[my_igp*ngpown+ig] * sexcut;
                    if(ssx > ssxcutoff && wx_array[iw] < 0.00) ssx = 0.00;

                    ssxt += matngmatmgp * ssx;
                    scht += matngmatmgp * sch;
                }
           }
        }
    }
}

void noflagOCC_solver(int *inv_igp_index, int *indinv, double *wx_array, double *wtilde_array, double *aqsmtemp, double *aqsntemp, double *I_eps_array, double *vcoul, int n1, int ngpown, int ncouls, double *achtemp)
{
#pragma omp parallel for  default(shared) 
    for(int my_igp=0; my_igp<ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];

        double achtemp_loc[nend-nstart];
        for(int iw = nstart; iw < nend; ++iw) {achtemp_loc[iw] = 0.00;}

        for(int ig = 0; ig<ncouls; ++ig)
        {
#pragma unroll(nend)
            for(int iw = nstart; iw < nend; ++iw)
            {
                double wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig]; //1 flops
                double delw = wtilde_array[my_igp*ncouls+ig] * wdiff * 1/(wdiff * wdiff); //4 ops 
                double sch_array = aqsmtemp[n1*ncouls+igp] * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[ig]; //5 ops
                achtemp_loc[iw] += sch_array; // 1 op
            }
        } //ncouls

        for(int iw = nstart; iw < nend; ++iw)
        {
#pragma omp atomic
            achtemp[iw] += achtemp_loc[iw];
        }
    } //ngpown
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);


    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;

    int inv_igp_index[ngpown];
    int indinv[ncouls+1];

    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;


    double to1 = 1e-6, \
    gamma = 0.5, \
    sexcut = 4.0;
    double limitone = 1.0/(to1*4.0), \
    limittwo = pow(0.5,2);

    double e_n1kq= 6.0; 

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
    
   
    double *acht_n1_loc = new double[number_bands];
    double *achtemp = new double[nend-nstart];
    double *asxtemp = new double[nend-nstart];
    double *aqsmtemp = new double[number_bands*ncouls];
    double *aqsntemp = new double[number_bands*ncouls];
    double *I_eps_array = new double[ngpown*ncouls];
    double *wtilde_array = new double[ngpown*ncouls];
    double *ssx_array = new double[3];
    double *ssxa = new double[ncouls];
    double achstemp;

    double *vcoul = new double[ncouls];
    double wx_array[nend-nstart];
    double occ=1.0;
    bool flag_occ;
    double achstemp_real = 0.00, achstemp_imag = 0.00;
    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;


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


    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0, tmp=1; ig<ncouls; ++ig,tmp++)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

       for(int iw=nstart; iw<nend; ++iw)
       {
           asxtemp[iw] = 0.00;
           achtemp[iw] = 0.00;
       }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

    flagOCC_solver(inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, nvband, ncouls, number_bands, ngpown, asxtemp);

#pragma omp parallel for 
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

    __SSC_MARK(0x111);
    __itt_resume();
    for(int n1 = 0; n1<number_bands; ++n1) 
        noflagOCC_solver(inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, n1, ngpown, ncouls, achtemp);
    __itt_pause();
    __SSC_MARK(0x222);

    //Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    printf(" \n Final achstemp\n");
    cout << "achstemp = " << achstemp << endl;

    printf("\n Final achtemp\n");
    cout << "achtemp[0] = " << achtemp[0] << endl;

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

    return 0;
}
