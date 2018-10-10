#include "./CustomComplex.h"
#define __OMPOFFLOAD__ 1

inline double elapsedTime(timeval start_time, timeval end_time)
{
    return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}

inline void schDttt_corKernel1(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

inline void schDttt_corKernel2(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

void calculate_schDt_lin3(CustomComplex<double>& schDt_lin3, CustomComplex<double>* sch2Di, bool flag_occ, int freqevalmin, double *ekq, int iw, int freqevalstep, double cedifft_zb_right, double cedifft_zb_left, CustomComplex<double> schDt_left, CustomComplex<double> schDt_lin2, int n1, double pref_zb, CustomComplex<double> pref_zb_compl, CustomComplex<double> schDt_avg)
{
    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
    if(intfact < 0.0001) intfact = 0.0001;
    if(intfact > 10000) intfact = 10000;
    intfact = -log(intfact);
    sch2Di[iw] = sch2Di[iw] - pref_zb_compl * schDt_avg * intfact;
    if(flag_occ)
    {
       double  intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
        if(intfact < 0.0001) intfact = 0.0001;
        if(intfact > 10000) intfact = 10000;
        intfact = log(intfact);
        schDt_lin3 = (schDt_left + schDt_lin2) * (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact ;
    }
    else
        schDt_lin3 = (schDt_left + schDt_lin2) * (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact;

}

static inline void compute_fact(double wx, int nFreq, double *dFreqGrid, double &fact1, double &fact2, int &ifreq, int loop, bool flag_occ)
{
    if(loop == 1 && wx > 0.00)
    {
            for(int ijk = 0; ijk < nFreq-1; ++ijk)
            {
                if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
            }
            if(ifreq == 0) ifreq = nFreq-2;
            fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
            fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    else if(loop == 1)
    {
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (-dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    if(loop == 2 && wx > 0.00)
    {
            for(int ijk = 0; ijk < nFreq-1; ++ijk)
            {
                if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                    ifreq = ijk;
            }
            if(ifreq == -1) ifreq = nFreq-2;
            fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
            fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
    }
    else if(loop == 2 && flag_occ)
    {
        wx = -wx; ifreq = 0;
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
        fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

    }
}

inline void ssxDittt_kernel(int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_eps_array, CustomComplex<double> &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, double fact1, double fact2, int igp, int my_igp)
{
    CustomComplex<double> ssxDitt(0.00, 0.00);
    for(int ig = 0; ig < ncouls; ++ig)
    {
        CustomComplex<double> ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                     I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

        ssxDitt += aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * ssxDit * vcoul[igp];
    }
    ssxDittt = ssxDitt;
}


void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int nFreq, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsntemp, CustomComplex<double> *aqsmtemp, CustomComplex<double> *I_epsR_array, double *vcoul, CustomComplex<double> &achsDtemp, double &elapsedTimeKernel)
{
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;
    timeval startTime, endTime;

    gettimeofday(&startTime, NULL);
#if __OMPOFFLOAD__

#pragma omp target teams distribute collapse(2) \
    map(to:inv_igp_index[0:ngpown], indinv[0:ncouls], aqsntemp[0:number_bands*ncouls], aqsmtemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls])\
    reduction(+:achsDtemp_re, achsDtemp_im)
//#pragma omp parallel for default(shared) collapse(2) reduction(+:achsDtemp_re, achsDtemp_im)
#else
#pragma omp parallel for default(shared) collapse(2) reduction(+:achsDtemp_re, achsDtemp_im)
#endif
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            int igblk = 1024;

#if __OMPOFFLOAD__
#pragma omp parallel for \
    reduction(+:achsDtemp_re, achsDtemp_im) 
            for(int ig = 0; ig < ncouls; ++ig)
            {
                CustomComplex<double> schsDtemp(0.00, 0.00);
                CustomComplex<double> aqsmtemp_conj = CustomComplex_conj(&aqsmtemp[n1*ncouls + igp]);
                CustomComplex<double> schsDtemp_store1 = CustomComplex_product(&aqsntemp[n1*ncouls + ig], &aqsmtemp_conj);
                CustomComplex<double> schsDtemp_store2 = CustomComplex_product(&schsDtemp_store1, &I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]);
                double schsDtemp_store_re = vcoul[igp] * 0.5;
                CustomComplex<double> schsDtemp_store3 = CustomComplex_product(&schsDtemp_store2, &schsDtemp_store_re);
                CustomComplex_minusEquals(&schsDtemp , &schsDtemp_store3);
                achsDtemp_re += CustomComplex_real(&schsDtemp);
                achsDtemp_im += CustomComplex_imag(&schsDtemp);
            }
#else
            CustomComplex<double> schsDtemp(0.00, 0.00);
            for(int igbeg = 0; igbeg < ncouls; igbeg+=igblk)
            {
                for(int ig = igbeg; ig < std::min(ncouls, igbeg+igblk); ++ig)
                    schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig];
            }
            achsDtemp_re += CustomComplex_real(schsDtemp * vcoul[igp] * 0.5);
            achsDtemp_im += CustomComplex_imag(schsDtemp * vcoul[igp] * 0.5);
#endif
        }
    } //n1

    gettimeofday(&endTime, NULL);
    elapsedTimeKernel = elapsedTime(startTime, endTime);

    achsDtemp = CustomComplex<double> (achsDtemp_re, achsDtemp_im) ;

}
//
inline void asxDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *asxDtemp, double &elapsedTimeKernel)
{
    double *asxDtemp_re = new double[nfreqeval];
    double *asxDtemp_im = new double[nfreqeval];
    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        asxDtemp_re[iw] = 0.00;
        asxDtemp_im[iw] = 0.00;
    }

    timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
#if __OMPOFFLOAD__
#pragma omp target teams distribute parallel for collapse(3) \
    map(to:inv_igp_index[0:ngpown], indinv[0:ncouls], aqsntemp[0:number_bands*ncouls], aqsmtemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls], \
    ekq[0:number_bands], dFreqGrid[0:nFreq])\
    map(tofrom : asxDtemp_re[0:nfreqeval], asxDtemp_im[0:nfreqeval])
#else
#pragma omp parallel for collapse(3) default(shared)
#endif 
    for(int n1 = 0; n1 < nvband; ++n1)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int iw = 0; iw < nfreqeval; ++iw)
            {
                double wx = freqevalmin - ekq[n1] + freqevalstep;
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                double fact1 = 0.00, fact2 = 0.00;
                int ifreq = 0;
                CustomComplex<double> ssxDittt(0.00, 0.00);

                compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

    //The ssxDittt_kernel is OMP parallelized.
            if(wx > 0)
                ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2, igp, my_igp);
                else
                    ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2, igp, my_igp);

                ssxDittt *= occ;
#pragma omp atomic
                asxDtemp_re[iw] += CustomComplex_real(ssxDittt );
#pragma omp atomic
                asxDtemp_im[iw] += CustomComplex_imag(ssxDittt );
            } // iw
        }
    }
    gettimeofday(&endTime, NULL);
    elapsedTimeKernel = elapsedTime(startTime, endTime);

    for(int iw = 0; iw < nfreqeval; ++iw)
        asxDtemp[iw] = CustomComplex<double>(asxDtemp_re[iw], asxDtemp_im[iw]);

    free(asxDtemp_re);
    free(asxDtemp_im);
}


void achDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double pref_zb, double *pref, double *dFreqGrid, CustomComplex<double> *dFreqBrd, CustomComplex<double> *schDt_matrix, CustomComplex<double> *schDi, CustomComplex<double> *schDi_cor, CustomComplex<double> *sch2Di, CustomComplex<double> *achDtemp)
{
    bool flag_occ;
    CustomComplex<double> expr0(0.00, 0.00);
#pragma omp parallel for default(shared) collapse(2)
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
        {
            flag_occ = n1 < nvband;
            CustomComplex<double> schDt = schDt_matrix[n1*nFreq + ifreq];
            double cedifft_zb = dFreqGrid[ifreq];
            double cedifft_zb_right, cedifft_zb_left;
            CustomComplex<double> schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            CustomComplex<double> cedifft_compl(cedifft_zb, 0.00);
            CustomComplex<double> cedifft_cor;
            CustomComplex<double> cedifft_coh = cedifft_compl - dFreqBrd[ifreq];
            CustomComplex<double> pref_zb_compl(0.00, pref_zb);

            if(flag_occ)
                cedifft_cor = cedifft_compl * -1 - dFreqBrd[ifreq];
                else
                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];

            if(ifreq != 0)
            {
                cedifft_zb_right = cedifft_zb;
                cedifft_zb_left = dFreqGrid[ifreq-1];
                schDt_right = schDt;
                schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
                schDt_avg = (schDt_right + schDt_left) * 0.5;
                schDt_lin = schDt_right - schDt_left;
                schDt_lin2 = schDt_lin / (cedifft_zb_right - cedifft_zb_left);

                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    sch2Di[iw] = expr0;
                    calculate_schDt_lin3(schDt_lin3, sch2Di, flag_occ, freqevalmin, ekq, iw, freqevalstep, cedifft_zb_right, cedifft_zb_left, schDt_left, schDt_lin2, n1, pref_zb, pref_zb_compl, schDt_avg);

                    schDt_lin3 += schDt_lin;
                    schDi_cor[iw] = schDi_cor[iw] -  (pref_zb_compl * schDt_lin3);
                }
            }

            for(int iw = 0; iw < nfreqeval; ++iw)
            {
                schDi[iw] = expr0;
                double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
                CustomComplex<double> tmp(0.00, pref[ifreq]);
                schDi[iw] = schDi[iw] - ((tmp*schDt) / (wx- cedifft_coh));
                achDtemp[iw] += schDi[iw];
            }
        }
    }

}

static inline void achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *achDtemp_cor, double &elapsedTimeKernel)
{
    timeval startTime, endTime;
    gettimeofday(&startTime, NULL);

    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
        schDttt_re = 0.00, schDttt_im = 0.00;
    double *achDtemp_cor_re = new double[nfreqeval];
    double *achDtemp_cor_im = new double[nfreqeval];
    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        achDtemp_cor_re[iw] = 0.00;
        achDtemp_cor_im[iw] = 0.00;
    }

#if __OMPOFFLOAD__
#pragma omp target teams distribute parallel for collapse(2) \
    map(to:inv_igp_index[0:ngpown], indinv[0:ncouls], aqsntemp[0:number_bands*ncouls], aqsmtemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls], \
    ekq[0:number_bands], dFreqGrid[0:nFreq])\
    map(tofrom:achDtemp_cor_re[0:nfreqeval], achDtemp_cor_im[0:nfreqeval])
#endif 
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            bool flag_occ = n1 < nvband;
            CustomComplex<double> sch2Di(0.00, 0.00);
            CustomComplex<double> schDi_cor(0.00, 0.00);
            CustomComplex<double> schDi_corb(0.00, 0.00);
            double wx = freqevalmin - ekq[n1] + freqevalstep;

            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0.00;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);

            if(wx > 0)
            {
                if(!flag_occ)
                schDttt_corKernel1(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                schDttt_corKernel2(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);


//Summing up at the end of iw loop
#pragma omp atomic
            achDtemp_cor_re[iw] += CustomComplex_real(schDi_cor);
#pragma omp atomic
            achDtemp_cor_im[iw] += CustomComplex_imag(schDi_cor);
        }// iw
    } //n1
    gettimeofday(&endTime, NULL);
    elapsedTimeKernel = elapsedTime(startTime, endTime);

    for(int iw = 0; iw < nfreqeval; ++iw)
        achDtemp_cor[iw] = CustomComplex<double>(achDtemp_cor_re[iw], achDtemp_cor_im[iw]);
}

inline void schDttt_corKernel1(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 512;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
    double schDttt_re = 0.00, schDttt_im = 0.00;
#if !__OMPOFFLOAD__
#pragma omp parallel for default(shared) collapse(2) reduction(+:schDttt_cor_re, schDttt_cor_im, schDttt_re, schDttt_im)
#endif
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                CustomComplex<double> sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
                CustomComplex<double> sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];


                schDttt_re += CustomComplex_real(sch2Dtt) ;
                schDttt_im += CustomComplex_imag(sch2Dtt) ;
#if !__OMPOFFLOAD__
#pragma omp atomic
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
#pragma omp atomic
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
#else 
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
#endif
            }
        }
    }
    schDttt_cor = CustomComplex<double> (schDttt_cor_re, schDttt_cor_im);

}

inline void schDttt_corKernel2(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 1024;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
#if !__OMPOFFLOAD__
#pragma omp parallel for default(shared) collapse(2) reduction(+:schDttt_cor_re, schDttt_cor_im)
#endif
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                CustomComplex<double> sch2Dt = ((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig]) * fact1 + \
                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2) * -0.5;
                CustomComplex<double> sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
#if !__OMPOFFLOAD__
#pragma omp atomic
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
#pragma omp atomic
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
#else 
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
#endif
            }
        }
    }
    schDttt_cor = CustomComplex<double> (schDttt_cor_re, schDttt_cor_im);
}

int main(int argc, char** argv)
{

    if(argc != 7)
    {
        cout << "Incorrect Parameters!!! The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int ngpown = atoi(argv[4]);
    const int nFreq = atoi(argv[5]);
    const int nfreqeval = atoi(argv[6]);

    const double freqevalmin = 0.00;
    const double freqevalstep = 0.50;
    const double occ = 1.00;
    const double pref_zb = 0.5 / 3.14;
    double dw = -10;

    if(ngpown > ncouls)
    {
        cout << "Incorrect Parameters!!! ngpown cannot be greater than ncouls. The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    timeval startTimer_Kernel, endTimer_Kernel, \
        start_achsDtemp_Kernel, end_achsDtemp_Kernel, \
        start_asxDtemp_Kernel, end_asxDtemp_Kernel, \
        start_achDtemp_Kernel, end_achDtemp_Kernel, 
        start_achDtemp_cor_Kernel, end_achDtemp_cor_Kernel, \
        start_preKernel, end_preKernel;

        double elapsed_achsDtemp = 0.00, elapsed_asxDtemp = 0.00, elapsed_achDtemp_cor = 0.00;
        
    gettimeofday(&start_preKernel, NULL);

#if __OMPOFFLOAD__
    cout << "OpenMP 4.5" << endl;
#else
    cout << "OpenMP 3.0" << endl;
#endif

    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

#if __OMPOFFLOAD__
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
#endif


    cout << "Number of Threads = " << numThreads << \
        "\n number_bands = " << number_bands << \
        "\n nvband = " << nvband << \
        "\n ncouls = " << ncouls << \
        "\n ngpown = " << ngpown << \
        "\n nFreq = " << nFreq << \
        "\n nfreqeval = " << nfreqeval << endl;

    CustomComplex<double> expr0( 0.0 , 0.0);
    CustomComplex<double> expr( 0.5 , 0.5);
    CustomComplex<double> expR( 0.5 , 0.5);
    CustomComplex<double> expA( 0.5 , -0.5);
    CustomComplex<double> exprP1( 0.5 , 0.1);

//Start to allocate the data structures;
    long double mem_alloc = 0.00;

    int *inv_igp_index = new int[ngpown];
    mem_alloc += ngpown * sizeof(int);

    int *indinv = new int[ncouls];
    mem_alloc += ncouls * sizeof(int);

    double *vcoul = new double[ncouls];
    mem_alloc += ncouls * sizeof(double);

    double *ekq = new double[number_bands];
    mem_alloc += number_bands * sizeof(double);

    double *dFreqGrid = new double[nFreq];
    double *pref = new double[nFreq];
    mem_alloc += 2 * nFreq * sizeof(double);

    CustomComplex<double> *aqsntemp = new CustomComplex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *aqsmtemp= new CustomComplex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *I_epsR_array = new CustomComplex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *I_epsA_array = new CustomComplex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *schDi = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *sch2Di = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *schDi_cor = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *schDi_corb = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *achDtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *achDtemp_cor = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *asxDtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *dFreqBrd = new CustomComplex<double>[nFreq];
    mem_alloc += (nfreqeval * 9 * sizeof(CustomComplex<double>));
    mem_alloc += (nFreq * sizeof(CustomComplex<double>)) ;

    CustomComplex<double> *schDt_matrix = new CustomComplex<double>[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(CustomComplex<double>));

    //Variables used : 
    CustomComplex<double> achsDtemp(0.00, 0.00);


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
        schDi[i] = expr0;
        sch2Di[i] = expr0;
        schDi_corb[i] = expr0;
        schDi_cor[i] = expr0;
        asxDtemp[i] = expr0;
        achDtemp[i] = expr0;
        achDtemp_cor[i] = expr0;
    }

    gettimeofday(&end_preKernel, NULL);
    double elapsed_preKernel = elapsedTime(start_preKernel, end_preKernel);

    cout << "pre kernel time taken = " << elapsed_preKernel << " secs" << endl;
    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;

#if __OMPOFFLOAD__
#pragma omp target enter data map(alloc: inv_igp_index[0:ngpown], indinv[0:ncouls], aqsntemp[0:number_bands*ncouls], aqsmtemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls])
#pragma omp target update to(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsntemp[0:number_bands*ncouls], aqsmtemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls])

    long double device_mem_alloc = 0.00;
    device_mem_alloc += 2*(number_bands * ncouls * sizeof(CustomComplex<double>));
    device_mem_alloc += 2*(nFreq * ngpown * ncouls * sizeof(CustomComplex<double>));
    device_mem_alloc += ngpown * sizeof(int);
    device_mem_alloc += ncouls * sizeof(int);
    device_mem_alloc += ncouls * sizeof(double);
    std::cout << "Memory Allocated on Device = " << device_mem_alloc/(1024 * 1024 * 1024) << " GBs" << std::endl;

#endif 

    gettimeofday(&startTimer_Kernel, NULL);
    cout << "starting Kernels" << endl;

    /***********achsDtemp Kernel ****************/
    achsDtemp_Kernel(number_bands, ngpown, ncouls, nFreq, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp, elapsed_achsDtemp);

    /***********asxDtemp Kernel ****************/
    asxDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp, elapsed_asxDtemp);

    /***********achDtemp Kernel ****************/
    achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, schDt_matrix, schDi, schDi_cor, sch2Di, asxDtemp);

    /***********achDtemp_cor Kernel ****************/
    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, achDtemp_cor, elapsed_achDtemp_cor);

    gettimeofday(&endTimer_Kernel, NULL);
    double elapsedTimer_Kernel = elapsedTime(startTimer_Kernel, endTimer_Kernel);

#if __OMPOFFLOAD__
#pragma omp target exit data map(delete: inv_igp_index[0:ngpown], indinv[0:ncouls], aqsntemp[0:number_bands*ncouls], aqsmtemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls])
#endif 
    cout << "achsDtemp = " ;
    achsDtemp.print();
    cout << "asxDtemp = " ;
    asxDtemp[0].print();
    cout << "achDtemp_cor = " ;
    achDtemp_cor[0].print();

    cout << "********** achsDtemp Time Taken **********= " << elapsed_achsDtemp << " secs" << endl;
    cout << "********** asxDtemp Time Taken **********= " << elapsed_asxDtemp << " secs" << endl;
    cout << "********** achDtemp_cor Time Taken **********= " << elapsed_achDtemp_cor << " secs" << endl;
    cout << "********** Kernel Time Taken **********= " << elapsedTimer_Kernel << " secs" << endl;

//Free the allocated memory 
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
    free(schDi);
    free(sch2Di);
    free(schDi_cor);
    free(schDi_corb);
    free(achDtemp);
    free(achDtemp_cor);
    free(asxDtemp);
    free(dFreqBrd);
    free(schDt_matrix);

    return 0;
}
