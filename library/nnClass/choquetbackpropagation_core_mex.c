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

long double sign(long double x)
{
    if (x>0)
        return 1.0;
    else if (x==0)
        return 0.0;
    else
        return -1.0;
}

void choquetbackpropagation_core_mex(double* choquetLayerCount, double* NETWORK_NODE_SIZE_TEMP, double* CHOQUET_NODE_SIZE_TEMP, double* CHOQUET_INPUT_SIZE, double* z_diff, double* LEARN_RATE, double* z_, double* dirac_mlp, const mxArray* inv_sigma, const mxArray* V, const mxArray* G, const mxArray* w, mxArray* V_new, mxArray* G_new)
{
    int i,j,k,l;
    
    unsigned int start_index, stop_index, cur_index;
    unsigned int CHOQUET_NODE_SIZE_SUM_1, CHOQUET_NODE_SIZE_SUM_2, CHOQUET_NODE_SIZE_SUM_3, CHOQUET_NODE_SIZE_SUM_ALL;
    
    mxArray* inv_sigma_temp;
    double* inv_sigma_temp_array;
    
    long double* x_org_array;
    long double* x_sigma_array;
    
    long double* sum_dir_V;
    
    mxArray* V_org;
    mxArray* V_org_temp;
    mxArray* V_new_temp;
    double* V_org_temp_array;
    double* V_new_temp_array;
    double* V_inv_sigma_array;
    long double V_abs_sum;
    V_org=mxDuplicateArray(V);
    
    mxArray* G_org;
    mxArray* G_org_temp;
    mxArray* G_new_temp;
    double* G_org_temp_array;
    double* G_new_temp_array;
    G_org=mxDuplicateArray(G);
    
    mxArray* w_temp;
    double* w_temp_array;
    
    CHOQUET_NODE_SIZE_SUM_ALL=0;
    for (k=1; k<=(*choquetLayerCount); k=k+1)
    {
        CHOQUET_NODE_SIZE_SUM_ALL=CHOQUET_NODE_SIZE_SUM_ALL+CHOQUET_NODE_SIZE_TEMP[k-1];
    }
    long double dirac[CHOQUET_NODE_SIZE_SUM_ALL];
    
    for (i=(*choquetLayerCount); i>=1; i=i-1)
    {
        CHOQUET_NODE_SIZE_SUM_1=0; CHOQUET_NODE_SIZE_SUM_2=0; CHOQUET_NODE_SIZE_SUM_3=0;
        for (k=1; k<=(i-2); k=k+1)
        {
            CHOQUET_NODE_SIZE_SUM_1=CHOQUET_NODE_SIZE_SUM_1+CHOQUET_NODE_SIZE_TEMP[k-1];
        }
        CHOQUET_NODE_SIZE_SUM_2=CHOQUET_NODE_SIZE_SUM_1;
        if ((i-1)>0)
        {
            CHOQUET_NODE_SIZE_SUM_2=CHOQUET_NODE_SIZE_SUM_2+CHOQUET_NODE_SIZE_TEMP[i-2];
        }
        CHOQUET_NODE_SIZE_SUM_3=CHOQUET_NODE_SIZE_SUM_2+CHOQUET_NODE_SIZE_TEMP[i-1];
        
        start_index=(*CHOQUET_INPUT_SIZE)*unitstep(i-2)+CHOQUET_NODE_SIZE_SUM_1;
        stop_index=(*CHOQUET_INPUT_SIZE)+CHOQUET_NODE_SIZE_SUM_2;

        sum_dir_V=mxCalloc(CHOQUET_NODE_SIZE_TEMP[i-1], sizeof(long double));
        
        if (i==(*choquetLayerCount))
        {
            for (k=1; k<=NETWORK_NODE_SIZE_TEMP[0]; k=k+1)
            {
                cur_index=k-1;
                w_temp=mxGetCell(w, cur_index);
                w_temp_array=mxGetData(w_temp);
                
                #pragma omp parallel for
                for (l=1;l<=CHOQUET_NODE_SIZE_TEMP[i-1];l=l+1)
                {
                    sum_dir_V[l-1]=sum_dir_V[l-1]+(long double)dirac_mlp[cur_index]*(long double)w_temp_array[l-1];
                }
            }
        }
        else
        {
            for (k=1; k<=CHOQUET_NODE_SIZE_TEMP[i]; k=k+1)
            {
                cur_index=CHOQUET_NODE_SIZE_SUM_3+k-1;
                V_org_temp=mxGetCell(V_org, cur_index);
                V_org_temp_array=mxGetData(V_org_temp);

                V_abs_sum=FLT_EPSILON;
                for (l=1;l<=CHOQUET_NODE_SIZE_TEMP[i-1];l=l+1)
                {
                    V_abs_sum=V_abs_sum+fabsl((long double)V_org_temp_array[l-1]);
                }
                #pragma omp parallel for
                for (l=1;l<=CHOQUET_NODE_SIZE_TEMP[i-1];l=l+1)
                {
                    sum_dir_V[l-1]=sum_dir_V[l-1]+dirac[cur_index]*fabsl((long double)V_org_temp_array[l-1])/V_abs_sum;
                }
            }
        }
        for (k=1; k<=CHOQUET_NODE_SIZE_TEMP[i-1]; k=k+1)
        {
            cur_index=CHOQUET_NODE_SIZE_SUM_2+k-1;

            dirac[cur_index]=sum_dir_V[k-1]*(long double)z_diff[stop_index+k-1];

            x_org_array=mxCalloc(stop_index-start_index, sizeof(long double));
            x_sigma_array=mxCalloc(stop_index-start_index, sizeof(long double));

            V_org_temp=mxGetCell(V_org, cur_index);
            V_org_temp_array=mxGetData(V_org_temp);
            V_new_temp=mxCreateDoubleMatrix(stop_index-start_index,1,mxREAL);
            V_new_temp_array=mxCalloc(stop_index-start_index, sizeof(double));
            V_inv_sigma_array=mxCalloc(stop_index-start_index, sizeof(double));
            G_org_temp=mxGetCell(G_org, cur_index);
            G_org_temp_array=mxGetData(G_org_temp);
            G_new_temp=mxCreateDoubleMatrix(stop_index-start_index,1,mxREAL);
            G_new_temp_array=mxCalloc(stop_index-start_index, sizeof(double));

            inv_sigma_temp=mxGetCell(inv_sigma, ((k-1)*(*choquetLayerCount)+i-1));
            inv_sigma_temp_array=mxGetData(inv_sigma_temp);

            #pragma omp parallel for
            for (l=0;l<=(stop_index-start_index-1);l=l+1)
                x_org_array[l]=(long double)z_[start_index+l]+(long double)G_org_temp_array[l];
            
            #pragma omp parallel for
            for (l=0;l<=(stop_index-start_index-1);l=l+1)
                x_sigma_array[l]=x_org_array[(unsigned int)inv_sigma_temp_array[l]-1];
            
            V_abs_sum=LDBL_EPSILON;
            for (l=0;l<CHOQUET_NODE_SIZE_TEMP[i-1];l=l+1)
                V_abs_sum=V_abs_sum+fabsl((long double)V_org_temp_array[l]);
            
            #pragma omp parallel for
            for (l=0;l<=(stop_index-start_index-1);l=l+1)
            {
                V_new_temp_array[l]=(long double)V_org_temp_array[l]+(long double)(*LEARN_RATE)*(long double)dirac[cur_index]*sign((long double)V_org_temp_array[l])*((long double)x_sigma_array[l]-(long double)z_[stop_index+k-1])/V_abs_sum;
                V_inv_sigma_array[(unsigned int)inv_sigma_temp_array[l]-1]=V_org_temp_array[l];
                if(V_new_temp_array[l]<0)
                    V_new_temp_array[l]=fabsl(V_new_temp_array[l]);
            }
            
            #pragma omp parallel for
            for (l=0;l<=(stop_index-start_index-1);l=l+1)
                G_new_temp_array[l]=(long double)G_org_temp_array[l]+(long double)(*LEARN_RATE)*(long double)dirac[cur_index]*fabsl((long double)V_inv_sigma_array[l])/V_abs_sum;
            
            mxFree(mxGetData(V_new_temp));
            mxSetData(V_new_temp, V_new_temp_array);
            mxDestroyArray(mxGetCell(V_new, cur_index));
            mxSetCell(V_new,cur_index,V_new_temp);

            mxFree(mxGetData(G_new_temp));
            mxSetData(G_new_temp, G_new_temp_array);
            mxDestroyArray(mxGetCell(G_new, cur_index));
            mxSetCell(G_new,cur_index,G_new_temp);

            mxFree(x_org_array);
            mxFree(x_sigma_array);
            mxFree(V_inv_sigma_array);
        }
        mxFree(sum_dir_V);
    }
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    omp_set_num_threads(omp_get_max_threads());
    
    //temporal variables
    unsigned int k;
    
    //define variables
    double* choquetLayerCount;
    double* NETWORK_NODE_SIZE_TEMP;
    double* CHOQUET_NODE_SIZE_TEMP;
    double* CHOQUET_INPUT_SIZE; 
    double* z_diff;
    double* LEARN_RATE;
    double* z_;
    double* dirac_mlp;
    
    const mxArray* inv_sigma;
    const mxArray* V;
    const mxArray* G;
    const mxArray* w;
    
    mxArray* V_new;
    mxArray* G_new;

    //set input pointers
    choquetLayerCount = mxGetPr(prhs[0]);
    NETWORK_NODE_SIZE_TEMP = mxGetPr(prhs[1]);
    CHOQUET_NODE_SIZE_TEMP = mxGetPr(prhs[2]);
    CHOQUET_INPUT_SIZE = mxGetPr(prhs[3]);
    z_diff = mxGetPr(prhs[4]);
    LEARN_RATE = mxGetPr(prhs[5]);
    z_ = mxGetPr(prhs[6]);
    dirac_mlp = mxGetPr(prhs[7]);
    
    inv_sigma=prhs[8];
    V = prhs[9];
    G = prhs[10];
    w = prhs[11];
    
    //set output pointers
    plhs[0] = mxCreateCellMatrix(mxGetM(V), mxGetN(V));
    V_new = plhs[0];
    
    plhs[1] = mxCreateCellMatrix(mxGetM(G), mxGetN(G));
    G_new = plhs[1];
    
    //call computational routine
    choquetbackpropagation_core_mex(choquetLayerCount, NETWORK_NODE_SIZE_TEMP, CHOQUET_NODE_SIZE_TEMP, CHOQUET_INPUT_SIZE, z_diff, LEARN_RATE, z_, dirac_mlp, inv_sigma, V, G, w, V_new, G_new);
}