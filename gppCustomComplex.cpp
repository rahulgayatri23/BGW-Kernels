/* CustomComplex version of gpp ported on GPU with OpenMP4.5 */
#include "CustomComplex.h"

#pragma omp declare target
void ngpownKernel(int *inv_igp_index, int *indinv, double *vcoul, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, double *wx_array, int n1, int ngpown, int ncouls, double &achtemp_re0, double &achtemp_re1, double &achtemp_re2, double &achtemp_im0, double &achtemp_im1, double &achtemp_im2);

#pragma omp end declare target


inline void ngpownKernel(int *inv_igp_index, int *indinv, double *vcoul, CustomComplex *wtilde_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, CustomComplex *I_eps_array, double *wx_array, int n1, int ngpown, int ncouls, double &achtemp_re0, double &achtemp_re1, double &achtemp_re2, double &achtemp_im0, double &achtemp_im1, double &achtemp_im2)
{
#pragma omp parallel for \
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2) 
    for(int my_igp=0; my_igp<ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        double achtemp_re_loc[nend-nstart];
        double achtemp_im_loc[nend-nstart];

        for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

#pragma omp simd
        for(int ig = 0; ig<ncouls; ++ig)
        {
#pragma unroll
            for(int iw = nstart; iw < nend; ++iw)
            {
                CustomComplex wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                CustomComplex delw = (wtilde_array[my_igp*ncouls+ig] * CustomComplex_conj(wdiff)) * 1/CustomComplex_real(wdiff * CustomComplex_conj(wdiff)); 
                CustomComplex sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];
                achtemp_re_loc[iw] += CustomComplex_real(sch_array);
                achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
            }
        }
        achtemp_re0 += achtemp_re_loc[0];
        achtemp_re1 += achtemp_re_loc[1];
        achtemp_re2 += achtemp_re_loc[2];
        achtemp_im0 += achtemp_im_loc[0];
        achtemp_im1 += achtemp_im_loc[1];
        achtemp_im2 += achtemp_im_loc[2];

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
    auto start_totalTime = std::chrono::high_resolution_clock::now();

    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int nodes_per_group = atoi(argv[4]);
    const int npes = 1; 
    const int ngpown = ncouls / (nodes_per_group * npes); 

    const double e_lk = 10;
    const double to1 = 1e-6, \
    gamma = 0.5, \
    sexcut = 4.0;
    const double limitone = 1.0/(to1*4.0), \
    limittwo = pow(0.5,2);
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
    CustomComplex *achtemp = new CustomComplex[nend-nstart];
    CustomComplex *aqsmtemp = new CustomComplex[number_bands*ncouls];
    CustomComplex *aqsntemp = new CustomComplex[number_bands*ncouls];
    CustomComplex *I_eps_array = new CustomComplex[ngpown*ncouls];
    CustomComplex *wtilde_array = new CustomComplex[ngpown*ncouls];
    CustomComplex achstemp;

    double *achtemp_re = new double[nend-nstart];
    double *achtemp_im = new double[nend-nstart];
    double *vcoul = new double[ncouls];
    double wx_array[nend-nstart];
    
    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;


    CustomComplex expr(0.5, 0.5);
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
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
       }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

    auto start_chrono_withDataMovement = std::chrono::high_resolution_clock::now();

    double achtemp_re0 = 0.00, achtemp_re1 = 0.00, achtemp_re2 = 0.00, \
        achtemp_im0 = 0.00, achtemp_im1 = 0.00, achtemp_im2 = 0.00;

#pragma omp target enter data map(alloc: aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1])

#pragma omp target update to(aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], wtilde_array[0:ngpown*ncouls])

    auto start_chrono = std::chrono::high_resolution_clock::now();

#pragma omp target teams distribute num_teams(number_bands) thread_limit(32) shared(vcoul, aqsntemp, aqsmtemp, I_eps_array) map(to:wx_array[nstart:nend], aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1])\
    map(tofrom:achtemp_re[nstart:nend], achtemp_im[nstart:nend], achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2) \
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2) 
    for(int n1 = 0; n1<number_bands; ++n1) 
    {
#pragma omp parallel for \
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2) 
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            double achtemp_re_loc[nend-nstart];
            double achtemp_im_loc[nend-nstart];

            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

#pragma omp simd
            for(int ig = 0; ig<ncouls; ++ig)
            {
                int iw = nstart;
                for(int iw = nstart; iw < nend; ++iw)
                {
                    CustomComplex wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                    CustomComplex delw = (wtilde_array[my_igp*ncouls+ig] * CustomComplex_conj(wdiff)) * 1/CustomComplex_real(wdiff * CustomComplex_conj(wdiff)); 
                    CustomComplex sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];
                    achtemp_re_loc[iw] += CustomComplex_real(sch_array);
                    achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
                }
            }
//            for(int iw = nstart; iw < nend; ++iw)
//            {
//#pragma omp atomic
//                achtemp_re[iw] += achtemp_re_loc[iw];
//#pragma omp atomic
//                achtemp_im[iw] += achtemp_im_loc[iw];
//            }
            achtemp_re0 += achtemp_re_loc[0];
            achtemp_re1 += achtemp_re_loc[1];
            achtemp_re2 += achtemp_re_loc[2];
            achtemp_im0 += achtemp_im_loc[0];
            achtemp_im1 += achtemp_im_loc[1];
            achtemp_im2 += achtemp_im_loc[2];

        } //ngpown
    } // number-bands

    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;

#pragma omp target exit data map(delete: aqsmtemp[:0],aqsntemp[:0], I_eps_array[:0], wtilde_array[:0], vcoul[:0], inv_igp_index[:0], indinv[:0])

    std::chrono::duration<double> elapsed_chrono_withDataMovement = std::chrono::high_resolution_clock::now() - start_chrono_withDataMovement;

    achtemp_re[0] = achtemp_re0;
    achtemp_re[1] = achtemp_re1;
    achtemp_re[2] = achtemp_re2;
    achtemp_im[0] = achtemp_im0;
    achtemp_im[1] = achtemp_im1;
    achtemp_im[2] = achtemp_im2;

    printf("\n Final achtemp\n");

    for(int iw=nstart; iw<nend; ++iw)
    {
        CustomComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;
    cout << "********** Kernel Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;
    cout << "********** Kernel+DataMov Time Taken **********= " << elapsed_chrono_withDataMovement.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(vcoul);

    return 0;
}
