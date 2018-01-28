#include <iostream>
#include <chrono>
#include <complex>
#include <omp.h>
#include "Complex.h"
#include <ittnotify.h>

using namespace std;

int main(int argc, char** argv)
{

    if(argc != 7)
    {
        cout << "Incorrect Parameters!!! The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    auto startTimer = std::chrono::high_resolution_clock::now();
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int ngpown = atoi(argv[4]);
    int nFreq = atoi(argv[5]);
    int nfreqeval = atoi(argv[6]);

    if(ngpown > ncouls)
    {
        cout << "Incorrect Parameters!!! ngpown cannot be greater than ncouls. The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    //OpenMP variables
    int tid, numThreads;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }

    cout << "Number of Threads = " << numThreads << \
        "\n number_bands = " << number_bands << \
        "\n nvband = " << nvband << \
        "\n ncouls = " << ncouls << \
        "\n ngpown = " << ngpown << \
        "\n nFreq = " << nFreq << \
        "\n nfreqeval = " << nfreqeval << endl;

    GPUComplex expr0( 0.0 , 0.0);
    GPUComplex expr( 0.5 , 0.5);
    GPUComplex expR( 0.5 , 0.5);
    GPUComplex expA( 0.5 , -0.5);
    GPUComplex exprP1( 0.5 , 0.1);
    double pref_zb = 0.5 / 3.14;

//Start to allocate the data structures;
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];
    double *vcoul = new double[ncouls];
    double *ekq = new double[number_bands];
    double *dFreqGrid = new double[nFreq];
    double *pref = new double[nFreq];
    long double mem_alloc = 0.00;

    GPUComplex *aqsntemp = new GPUComplex[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    GPUComplex *aqsmtemp= new GPUComplex[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    GPUComplex *I_epsR_array = new GPUComplex[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    GPUComplex *I_epsA_array = new GPUComplex[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    GPUComplex *ssxDi = new GPUComplex[nfreqeval];
    GPUComplex *schDi = new GPUComplex[nfreqeval];
    GPUComplex *sch2Di = new GPUComplex[nfreqeval];
    GPUComplex *schDi_cor = new GPUComplex[nfreqeval];
    GPUComplex *schDi_corb = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp = new GPUComplex[nfreqeval];
    GPUComplex *ach2Dtemp = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp_cor = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp_corb = new GPUComplex[nfreqeval];
    GPUComplex *asxDtemp = new GPUComplex[nfreqeval];
    GPUComplex *dFreqBrd = new GPUComplex[nFreq];
    mem_alloc += (nfreqeval * 10 * sizeof(GPUComplex));
    mem_alloc += (nFreq * sizeof(GPUComplex)) ;

    GPUComplex *schDt_matrix = new GPUComplex[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(GPUComplex));

    //Variables used : 
    GPUComplex achsDtemp = expr0;
    GPUComplex *achsDtemp_threadArr = new GPUComplex[numThreads];
    GPUComplex *schDttt_cor_threadArr = new GPUComplex[numThreads];
    for(int i = 0; i < numThreads; ++i)
    {
        achsDtemp_threadArr[i] = expr0;
        schDttt_cor_threadArr[i] = expr0;
    }


    double freqevalmin = 0.00;
    double freqevalstep = 0.50;
    double dw = -10;

    //Initialize the data structures
    for(int ig = 0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig;

    for(int ig = 0; ig < ncouls; ++ig)
        indinv[ig] = ig;

    for(int i=0; i<number_bands; ++i)
    {
        ekq[i] = dw;
        dw += 1.00;

        for(int j=0; j<ncouls; ++j)
        {
            aqsmtemp[i*ncouls+j] = expr;
            aqsntemp[i*ncouls+j] = expr;
        }

        for(int j=0; j<nFreq; ++j)
            schDt_matrix[i*nFreq + j] = expr0;
    }

    for(int i=0; i<ncouls; ++i)
        vcoul[i] = 1.00;

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                I_epsR_array[i*ngpown*ncouls + j * ncouls + k] = expR;
                I_epsA_array[i*ngpown*ncouls + j * ncouls + k] = expA;
            }
        }
    }

    dw = 0.00;
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqBrd[ijk] = exprP1;
        dFreqGrid[ijk] = dw;
        dw += 2.00;
    }

    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq-1)
            pref[ifreq] = (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]) / 3.14;
            else
                pref[ifreq] = pref[ifreq-1];

    }
    pref[0] *= 0.5; pref[nFreq-1] *= 0.5;

    for(int i = 0; i < nfreqeval; ++i)
    {
        ssxDi[i] = expr0;
        schDi[i] = expr0;
        sch2Di[i] = expr0;
        schDi_corb[i] = expr0;
        schDi_cor[i] = expr0;
        asxDtemp[i] = expr0;
        achDtemp[i] = expr0;
        ach2Dtemp[i] = expr0;
        achDtemp_cor[i] = expr0;
        achDtemp_corb[i] = expr0;
    }
    GPUComplex *ssxDittt = new GPUComplex[numThreads];

    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;
    std::chrono::duration<double> elapsedTime_preloop = std::chrono::high_resolution_clock::now() - startTimer;

    cout << "starting loop" << endl;
    auto startTimer_firstloop = std::chrono::high_resolution_clock::now();
    auto startTimer_kernel = std::chrono::high_resolution_clock::now();

    for(int n1 = 0; n1 < nvband; ++n1)
    {
        double occ = 1.00;
        GPUComplex ssxDit = expr0;
        GPUComplex ssxDittt_agg = expr0;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
                for(int i = 0; i < numThreads; ++i)
                    ssxDittt[i] = expr0;
            double wx = freqevalmin - ekq[n1] + freqevalstep;
            ssxDi[iw] = expr0;
            GPUComplex ssxDittt_tmp = expr0;

            int ifreq = 0;
            if(wx > 0.00)
            {
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                    ifreq = ijk;
                }
            }
            else
            {
                int ifreq = 0;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                        ifreq = ijk;
                }
            }
            if(ifreq == 0) ifreq = nFreq-2;

            if(wx > 0.00)
            {
                double fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                double fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);


#pragma omp parallel for default(shared)
                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int indigp = inv_igp_index[my_igp];
                    int igp = indinv[indigp];
                    int igmax = ncouls;
                    GPUComplex ssxDitt = expr0;
                    int tid = omp_get_thread_num();

//                    if(igp < ncouls && igp >= 0)
                    {
                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            ssxDit = GPUComplex_mult(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) + \
                                                         GPUComplex_mult(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2);
        
                            ssxDitt += GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit));
                        }
                        ssxDittt[tid] += GPUComplex_mult(ssxDitt , vcoul[igp]);
                    }
                }
            }
            else
            {
                double fact1 = (dFreqGrid[ifreq+1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                double fact2 = (-dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                ssxDittt_tmp = expr0;

                for(int i = 0; i < numThreads; ++i)
                    ssxDittt[i] = expr0;

#pragma omp parallel for default(shared)
                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int indigp = inv_igp_index[my_igp];
                    int igp = indinv[indigp];
                    int igmax = ncouls;
                    GPUComplex ssxDitt = expr0;
                    int tid = omp_get_thread_num();

//                    if(igp < ncouls && igp >= 0)
                    {
                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            ssxDit = GPUComplex_mult(I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) + \
                                                         GPUComplex_mult(I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2);
        
                            ssxDitt += GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit));
                        }
                        ssxDittt[tid] += GPUComplex_mult(ssxDitt , vcoul[igp]);
                    }
                }
            }
                for(int i = 0; i < numThreads; ++i)
                    ssxDittt_tmp += ssxDittt[i];

            ssxDi[iw] += ssxDittt_tmp;
            asxDtemp[iw] += GPUComplex_mult(ssxDi[iw] , occ);
        } // iw
    }

    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;
        double occ = 1.00;

#pragma omp parallel for default(shared)
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            int tid = omp_get_thread_num();

//            if(igp < ncouls && igp >= 0)
            {
                int igmax = ncouls;
                GPUComplex schsDtemp = expr0;

                for(int ig = 0; ig < igmax; ++ig)
                    schsDtemp = schsDtemp - GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]);

                achsDtemp_threadArr[tid] += GPUComplex_mult(schsDtemp , vcoul[igp] * 0.5);
            }
        }
    } //n1

    for(int i = 0; i < numThreads; ++i)
        achsDtemp += achsDtemp_threadArr[i];

    std::chrono::duration<double> elapsedTime_firstloop = std::chrono::high_resolution_clock::now() - startTimer_firstloop;

//    /******************************Done with the First Part of the Code*****************************************************************************/

    auto startTimer_secondloop = std::chrono::high_resolution_clock::now();

    __SSC_MARK(0x111);
    __itt_resume();
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;
        double occ = 1.00;
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            schDi[iw] = expr0;
            sch2Di[iw] = expr0;
            schDi_corb[iw] = expr0;
            schDi_cor[iw] = expr0;
        }

#pragma omp parallel for default(shared)
        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
        {
            GPUComplex schDt = schDt_matrix[n1*nFreq + ifreq];
            double cedifft_zb = dFreqGrid[ifreq];
            double cedifft_zb_right, cedifft_zb_left;
            GPUComplex schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            GPUComplex cedifft_compl(cedifft_zb, 0.00);
            GPUComplex cedifft_cor;
            GPUComplex cedifft_coh = cedifft_compl - dFreqBrd[ifreq];

            if(flag_occ)
                cedifft_cor = GPUComplex_mult(cedifft_compl, -1) - dFreqBrd[ifreq];
                else
                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];

            if(ifreq != 0)
            {
                cedifft_zb_right = cedifft_zb;
                cedifft_zb_left = dFreqGrid[ifreq-1];
                schDt_right = schDt;
                schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
                schDt_avg = GPUComplex_mult((schDt_right + schDt_left) , 0.5);
                schDt_lin = schDt_right - schDt_left;
                schDt_lin2 = GPUComplex_divide(schDt_lin , (cedifft_zb_right - cedifft_zb_left));
            }

            if(ifreq != nFreq)
            {
                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
                    GPUComplex tmp(0.00, pref[ifreq]);
                    schDi[iw] = schDi[iw] - GPUComplex_divide(GPUComplex_product(tmp,schDt) , doubleMinusGPUComplex(wx, cedifft_coh));
                }
            }

            if(ifreq != 0)
            {
                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
                    if(intfact < 0.0001) intfact = 0.0001;
                    if(intfact > 10000) intfact = 10000;
                    intfact = -log(intfact);
                    GPUComplex pref_zb_compl(0.00, pref_zb);
                    sch2Di[iw] = sch2Di[iw] - GPUComplex_mult(GPUComplex_product(pref_zb_compl , schDt_avg) , intfact);
                    if(flag_occ)
                    {
                        intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
                        if(intfact < 0.0001) intfact = 0.0001;
                        if(intfact > 10000) intfact = 10000;
                        intfact = log(intfact);
                        schDt_lin3 = GPUComplex_mult((schDt_left + schDt_lin2) , (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact) ;
                    }
                    else
                        schDt_lin3 = GPUComplex_mult((schDt_left + schDt_lin2) , (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact);

                    schDt_lin3 += schDt_lin;
                    schDi_cor[iw] = schDi_cor[iw] -  GPUComplex_product(pref_zb_compl , schDt_lin3);
                }
            }
        }

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + freqevalstep;
            GPUComplex schDttt_cor = expr0;
            for(int i = 0; i < numThreads; ++i)
                schDttt_cor_threadArr[i] = expr0;

            if(wx > 0)
            {
                int ifreq = -1;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                        ifreq = ijk;
                }
                if(ifreq == -1) ifreq = nFreq-2;

                double fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
                double fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

                GPUComplex schDttt = expr0;

#pragma omp parallel for default(shared)
                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int tid = omp_get_thread_num();
                    int indigp = inv_igp_index[my_igp] ;
                    int igp = indinv[indigp];
                    int igmax = ncouls;
                    GPUComplex sch2Dtt(0.00, 0.00);

//                    if(igp < ncouls && igp >= 0)
                    {
                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            GPUComplex sch2Dt = GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig])) , fact1) + \
                                                        GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2);
                            sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt);
                        }
                        schDttt += GPUComplex_mult(sch2Dtt , vcoul[igp]);
                        if(flag_occ){}
                        else
                            schDttt_cor_threadArr[tid] += GPUComplex_mult(sch2Dtt , vcoul[igp]);
                    }
                }

                sch2Di[iw] += schDttt;
            }
            else if(flag_occ)
            {
                wx = -wx; int ifreq = 0;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                        ifreq = ijk;
                }
                if(ifreq == 0) ifreq = nFreq-2;

                double fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
                double fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

#pragma omp parallel for default(shared)
                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int tid = omp_get_thread_num();
                    int indigp = inv_igp_index[my_igp] ;
                    int igp = indinv[indigp];
                    int igmax = ncouls;
                    GPUComplex sch2Dtt(0.00, 0.00);

//                    if(igp < ncouls && igp >= 0)
                    {
                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            GPUComplex sch2Dt = GPUComplex_mult(GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig])) , fact1) + \
                                                        GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2), -0.5);
                            sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt);
                        }
                        schDttt_cor_threadArr[tid] += GPUComplex_mult(sch2Dtt , vcoul[igp]);
                    }
                }
            }

            for(int i = 0; i < numThreads; ++i)
                schDttt_cor += schDttt_cor_threadArr[i];

            schDi_cor[iw] += schDttt_cor;

//Summing up at the end of iw loop
            achDtemp[iw] += schDi[iw];
            ach2Dtemp[iw] += sch2Di[iw];
            achDtemp_cor[iw] += schDi_cor[iw];
            achDtemp_corb[iw] += schDi_corb[iw];
        }// iw
    } //n1

    __SSC_MARK(0x222);
    __itt_pause();
    std::chrono::duration<double> elapsedTime_secondloop = std::chrono::high_resolution_clock::now() - startTimer_secondloop;

    cout << "achsDtemp = " ;
    achsDtemp.print();
    cout << "asxDtemp = " ;
    asxDtemp[0].print();
    cout << "achDtemp_cor = " ;
    achDtemp_cor[0].print();

    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
    std::chrono::duration<double> elapsedKernelTime = std::chrono::high_resolution_clock::now() - startTimer_kernel;
    cout << "********** PreLoop **********= " << elapsedTime_preloop.count() << " secs" << endl;
    cout << "********** Kenel Time **********= " << elapsedKernelTime.count() << " secs" << endl;
//    cout << "********** FirtLoop **********= " << elapsedTime_firstloop.count() << " secs" << endl;
//    cout << "********** SecondLoop  **********= " << elapsedTime_secondloop.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

//Free the allocated memory since you are a good programmer :D
    free(aqsntemp);
    free(aqsmtemp);
    free(I_epsA_array);
    free(I_epsR_array);
    free(inv_igp_index);
    free(indinv);
    free(vcoul);
    free(ekq);
    free(dFreqGrid);
    free(pref);
    free(ssxDi);
    free(schDi);
    free(sch2Di);
    free(schDi_cor);
    free(schDi_corb);
    free(achDtemp);
    free(ach2Dtemp);
    free(achDtemp_cor);
    free(achDtemp_corb);
    free(asxDtemp);
    free(dFreqBrd);
    free(schDt_matrix);

    return 0;
}
