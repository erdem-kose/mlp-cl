#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <omp.h>
#include <openacc.h>

double unitstep(double x)
{
    if (x>=0)
        return 1.0;
    else
        return 0.0;
}

void backpropagation_core_mex(double* layerCount, double* NETWORK_NODE_SIZE_TEMP, double* MLP_INPUT_SIZE, double* e, double* x_diff, double* LEARN_RATE, double* x_, const mxArray* w, mxArray* w_new, double* dirac)
{
    int i,j,k,l;
    
    unsigned int start_index, stop_index, cur_index;
    unsigned int NETWORK_NODE_SIZE_SUM_1, NETWORK_NODE_SIZE_SUM_2, NETWORK_NODE_SIZE_SUM_3;
    
    long double* sum_dir_w;
    
    mxArray* w_org;
    mxArray* w_org_temp;
    mxArray* w_new_temp;
    double* w_org_temp_array;
    double* w_new_temp_array;
    
    w_org=mxDuplicateArray(w);

    for (i=(*layerCount); i>=1; i=i-1)
    {
        NETWORK_NODE_SIZE_SUM_1=0; NETWORK_NODE_SIZE_SUM_2=0; NETWORK_NODE_SIZE_SUM_3=0;
        for (k=1; k<=(i-2); k=k+1)
        {
            NETWORK_NODE_SIZE_SUM_1=NETWORK_NODE_SIZE_SUM_1+NETWORK_NODE_SIZE_TEMP[k-1];
        }
        NETWORK_NODE_SIZE_SUM_2=NETWORK_NODE_SIZE_SUM_1;
        if ((i-1)>0)
        {
            NETWORK_NODE_SIZE_SUM_2=NETWORK_NODE_SIZE_SUM_2+NETWORK_NODE_SIZE_TEMP[i-2];
        }
        NETWORK_NODE_SIZE_SUM_3=NETWORK_NODE_SIZE_SUM_2+NETWORK_NODE_SIZE_TEMP[i-1];

        start_index=(*MLP_INPUT_SIZE+i-1)*unitstep(i-2)+NETWORK_NODE_SIZE_SUM_1;
        stop_index=(*MLP_INPUT_SIZE+i)+NETWORK_NODE_SIZE_SUM_2;
        
        if (i==(*layerCount))
        {
            for (k=1; k<=NETWORK_NODE_SIZE_TEMP[i-1]; k=k+1)
            {
                cur_index=NETWORK_NODE_SIZE_SUM_2+k-1;
                dirac[cur_index]=(long double)e[k-1]*(long double)x_diff[stop_index+k-1];
                w_org_temp=mxGetCell(w_org, cur_index);
                w_org_temp_array=mxGetData(w_org_temp);
                w_new_temp=mxCreateDoubleMatrix(1,stop_index-start_index,mxREAL);
                w_new_temp_array=mxCalloc(stop_index-start_index, sizeof(double));
                #pragma omp parallel for
                for (l=(start_index+1);l<=stop_index;l=l+1)
                {
                    w_new_temp_array[l-(start_index+1)]=(long double)w_org_temp_array[l-(start_index+1)]+(long double)(*LEARN_RATE)*(long double)dirac[cur_index]*(long double)x_[l-1];
                }
               
                mxFree(mxGetData(w_new_temp));
                mxSetData(w_new_temp, w_new_temp_array);
                mxDestroyArray(mxGetCell(w_new, cur_index));
                mxSetCell(w_new,cur_index,w_new_temp);
            }
        }
        else
        {
            sum_dir_w=mxCalloc(NETWORK_NODE_SIZE_TEMP[i-1], sizeof(long double));

            for (k=1; k<=NETWORK_NODE_SIZE_TEMP[i]; k=k+1)
            {
                cur_index=NETWORK_NODE_SIZE_SUM_3+k-1;
                w_org_temp=mxGetCell(w_org, cur_index);
                w_org_temp_array=mxGetData(w_org_temp);
                #pragma omp parallel for
                for (l=1;l<=NETWORK_NODE_SIZE_TEMP[i-1];l=l+1)
                {
                    sum_dir_w[l-1]=(long double)sum_dir_w[l-1]+(long double)dirac[cur_index]*(long double)w_org_temp_array[l-1];
                }
            }
            for (k=1; k<=NETWORK_NODE_SIZE_TEMP[i-1]; k=k+1)
            {
                cur_index=NETWORK_NODE_SIZE_SUM_2+k-1;
                dirac[cur_index]=(long double)sum_dir_w[k-1]*(long double)x_diff[stop_index+k-1];
                w_org_temp=mxGetCell(w_org, cur_index);
                w_org_temp_array=mxGetData(w_org_temp);
                w_new_temp=mxCreateDoubleMatrix(1,stop_index-start_index,mxREAL);
                w_new_temp_array=mxCalloc(stop_index-start_index, sizeof(double));
                #pragma omp parallel for
                for (l=(start_index+1);l<=stop_index;l=l+1)
                {
                    w_new_temp_array[l-(start_index+1)]=(long double)w_org_temp_array[l-(start_index+1)]+(long double)(*LEARN_RATE)*(long double)dirac[cur_index]*(long double)x_[l-1];
                }
                mxFree(mxGetData(w_new_temp));
                mxSetData(w_new_temp, w_new_temp_array);
                mxDestroyArray(mxGetCell(w_new, cur_index));
                mxSetCell(w_new,cur_index,w_new_temp);
            }
            mxFree(sum_dir_w);
        }
    }
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    omp_set_num_threads(omp_get_max_threads());
    
    //temporal variables
    unsigned int NETWORK_NODE_SIZE_SUM_ALL,k;
    
    //define variables
    double* layerCount;
    double* NETWORK_NODE_SIZE_TEMP;
    double* MLP_INPUT_SIZE; 
    double* e;
    double* x_diff;
    double* LEARN_RATE;
    double* x_;
    double* dirac;
    const mxArray* w;
    mxArray* w_new;

    //set input pointers
    layerCount = mxGetPr(prhs[0]);
    NETWORK_NODE_SIZE_TEMP = mxGetPr(prhs[1]);
    MLP_INPUT_SIZE = mxGetPr(prhs[2]);
    e = mxGetPr(prhs[3]);
    x_diff = mxGetPr(prhs[4]);
    LEARN_RATE = mxGetPr(prhs[5]);
    x_ = mxGetPr(prhs[6]);
    w = prhs[7];
    
    //set output pointers
    plhs[0] = mxCreateCellMatrix(mxGetM(w), mxGetN(w));
    w_new = plhs[0];
    
    NETWORK_NODE_SIZE_SUM_ALL=0;
    for (k=1; k<=(*layerCount); k=k+1)
    {
        NETWORK_NODE_SIZE_SUM_ALL=NETWORK_NODE_SIZE_SUM_ALL+NETWORK_NODE_SIZE_TEMP[k-1];
    }
    plhs[1] = mxCreateDoubleMatrix(NETWORK_NODE_SIZE_SUM_ALL,1,mxREAL);
    dirac = mxGetPr(plhs[1]);
    
    //call computational routine
    backpropagation_core_mex(layerCount, NETWORK_NODE_SIZE_TEMP, MLP_INPUT_SIZE, e, x_diff, LEARN_RATE, x_, w, w_new, dirac);
}