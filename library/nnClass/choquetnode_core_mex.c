#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <omp.h>
#include <openacc.h>

//mexPrintf("%f %f \n",CHOQUET_NODE_SIZE_TEMP[0], CHOQUET_NODE_SIZE_TEMP[1]);

long double *base_arr;

static int compar (const void *a, const void *b)
{
    unsigned int aa = *((unsigned int *) a), bb = *((unsigned int *) b);
    if (base_arr[aa] < base_arr[bb])
        return -1;
    else if (base_arr[aa] == base_arr[bb])
        return 0;
    else if (base_arr[aa] > base_arr[bb])
        return 1;
}

double unitstep(double x)
{
    if (x>=0)
        return 1.0;
    else
        return 0.0;
}

void z_init(double* z_new, double* z, double* CHOQUET_INPUT_SIZE, mwSize m, mwSize n)
{
    mwSize i,j,m_raw;
    
    m_raw=(unsigned int)*CHOQUET_INPUT_SIZE;

    #pragma omp parallel for
    for (i=0;i<m_raw;i=i+1)
    {
        for (j=0;j<n;j=j+1)
        {
            z_new[(j*m)+i]=z[(j*m_raw)+i];
        }
    }
}

void choquet_mex(double *z, double *z_diff, double* inv_sigma_out, double *V, double *G, double *x, double alpha, mwSize m, mwSize n)
{
    mwSize i,j;
    unsigned int inv_sigma[m];
    long double v[n];
    long double xG[m];
    long double V_abs_sum;

    #pragma omp parallel for
    for (i=0; i<n; i=i+1) {
        v[i]=0.0;
        V_abs_sum=FLT_EPSILON;

        for (j=0; j<m; j=j+1) {
            xG[j] = (long double)G[j] + (long double)x[(i*m)+j];
            inv_sigma[j]=j;
            V_abs_sum=V_abs_sum+fabsl((long double)V[j]);
        }
        base_arr = xG;
        qsort(inv_sigma,m, sizeof(unsigned int),compar);
        for (j=0; j<m; j=j+1) {
            v[i] = v[i] + ((long double)fabsl(V[j])/V_abs_sum)*xG[inv_sigma[j]];
            inv_sigma_out[(i*m)+j]=(long double)(inv_sigma[j]+1);
        }

        z[i]=(long double)v[i];
        z_diff[i]=1.0;
//        
//         if (v[i]>=0)
//         {
//             z[i]=(long double)v[i];
//             z_diff[i]=1.0;
//         }
//         else
//         {
//             z[i]=(long double)alpha*(exp((long double)v[i])-1.0);
//             z_diff[i]=(long double)z[i]+(long double)alpha;
//         }
    }
}

void choquetnode_core_mex(double* choquetLayerCount, double* CHOQUET_NODE_SIZE_TEMP, double* CHOQUET_INPUT_SIZE, double hid_alpha, const mxArray* V, const mxArray* G, double* x_diff_out, double* x_out, double* z_out, mxArray* inv_sigma_out, mwSize m_x, mwSize m_x_max, mwSize m_y, mwSize n)
{
    int i,j,k,l;
    mwSize m;
    
    unsigned int start_index, stop_index, cur_index;
    unsigned int CHOQUET_NODE_SIZE_SUM_1, CHOQUET_NODE_SIZE_SUM_2, CHOQUET_NODE_SIZE_SUM_3;
    
    mxArray* V_temp;
    mxArray* G_temp;
    mxArray* inv_sigma_out_temp;
    double* V_temp_array;
    double* G_temp_array;
    double* inv_sigma_out_temp_array;
    
    double* x_temp;
    double* x_diff_out_temp;
    double* x_out_temp;
    x_out_temp=mxCalloc(n, sizeof(double));
    x_diff_out_temp=mxCalloc(n, sizeof(double));
    
    for (i=1; i<=(*choquetLayerCount); i=i+1)
    {
        CHOQUET_NODE_SIZE_SUM_1=0; CHOQUET_NODE_SIZE_SUM_2=0;
        for (k=1; k<=(i-2); k=k+1)
        {
            CHOQUET_NODE_SIZE_SUM_1=CHOQUET_NODE_SIZE_SUM_1+CHOQUET_NODE_SIZE_TEMP[k-1];
        }
        CHOQUET_NODE_SIZE_SUM_2=CHOQUET_NODE_SIZE_SUM_1;
        if ((i-1)>0)
        {
            CHOQUET_NODE_SIZE_SUM_2=CHOQUET_NODE_SIZE_SUM_2+CHOQUET_NODE_SIZE_TEMP[i-2];
        }
        start_index=(*CHOQUET_INPUT_SIZE)*unitstep(i-2)+CHOQUET_NODE_SIZE_SUM_1;
        stop_index=(*CHOQUET_INPUT_SIZE)+CHOQUET_NODE_SIZE_SUM_2;
        m=stop_index-start_index;

        x_temp=mxCalloc(m*n, sizeof(double));
        inv_sigma_out_temp_array=mxCalloc(m*n, sizeof(double));
        inv_sigma_out_temp=mxCreateDoubleMatrix(m,n,mxREAL);

        #pragma omp parallel for
        for (j=0; j<n; j=j+1)
        {
            #pragma omp parallel for
            for (l=(start_index+1);l<=(stop_index);l=l+1)
            {
                x_temp[(j*m)+l-(start_index+1)]=x_out[(j*m_x)+l-1];
            }
        }
        
        for (k=1; k<=CHOQUET_NODE_SIZE_TEMP[i-1]; k=k+1)
        {
            cur_index=CHOQUET_NODE_SIZE_SUM_2+k-1;
            V_temp=mxGetCell(V, cur_index);
            G_temp=mxGetCell(G, cur_index);
            V_temp_array=mxGetData(V_temp);
            G_temp_array=mxGetData(G_temp);
            
            inv_sigma_out_temp_array=mxCalloc(m*n, sizeof(double));
            inv_sigma_out_temp=mxCreateDoubleMatrix(m,n,mxREAL);
            
            choquet_mex(x_out_temp, x_diff_out_temp, inv_sigma_out_temp_array, V_temp_array, G_temp_array, x_temp, hid_alpha, m, n);
            
            mxFree(mxGetData(inv_sigma_out_temp));
            mxSetData(inv_sigma_out_temp, inv_sigma_out_temp_array);
            mxDestroyArray(mxGetCell(inv_sigma_out, ((k-1)*(*choquetLayerCount)+i-1)));
            mxSetCell(inv_sigma_out,((k-1)*(*choquetLayerCount)+i-1),inv_sigma_out_temp);

            #pragma omp parallel for
            for (j=0; j<n; j=j+1)
            {
                x_out[(j*m_x)+(stop_index+k-1)]=x_out_temp[j];
                x_diff_out[(j*m_x)+(stop_index+k-1)]=x_diff_out_temp[j];
                if (i==(*choquetLayerCount))
                    z_out[(j*m_y)+k-1]=x_out_temp[j];
            }
        }
        mxFree(x_temp);
    }
    mxFree(x_out_temp);
    mxFree(x_diff_out_temp);
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    omp_set_num_threads(omp_get_max_threads());
    
    mwSize i,m_x,m_x_max,m_y,n;
    //define variables
    double* choquetLayerCount;
    double* CHOQUET_NODE_SIZE_TEMP;
    double hid_alpha;
    double* CHOQUET_INPUT_SIZE;
    
    const mxArray* V;
    const mxArray* G;
    
    double* x_in;
    double* x_diff_out;
    double* x_out;
    double* z_out;
    mxArray* inv_sigma_out;
    
    //set input pointers
    choquetLayerCount = mxGetPr(prhs[0]);
    CHOQUET_NODE_SIZE_TEMP = mxGetPr(prhs[1]);
    CHOQUET_INPUT_SIZE = mxGetPr(prhs[2]);
    hid_alpha = mxGetScalar(prhs[3]);
    V = prhs[4];
    G = prhs[5];
    x_in=mxGetPr(prhs[6]);
    
    m_x=mxGetM(prhs[6]);
    m_y=(unsigned int)(CHOQUET_NODE_SIZE_TEMP[(unsigned int)(*choquetLayerCount-1)]);
    n=mxGetN(prhs[6]);
    
    //set output pointers
    plhs[0] = mxCreateDoubleMatrix(m_y,n,mxREAL);
    z_out = mxGetPr(plhs[0]);
    
    m_x_max=0;
    for (i=0;i<*choquetLayerCount;i=i+1)
    {
        m_x=m_x+CHOQUET_NODE_SIZE_TEMP[i];
        if (CHOQUET_NODE_SIZE_TEMP[i]>m_x_max)
            m_x_max=CHOQUET_NODE_SIZE_TEMP[i];
    }
    
    plhs[1] = mxCreateDoubleMatrix(m_x,n,mxREAL);
    x_out=mxGetPr(plhs[1]);
    
    z_init(x_out, x_in, CHOQUET_INPUT_SIZE, m_x, n);
    
    plhs[2] = mxCreateDoubleMatrix(m_x,n,mxREAL);
    x_diff_out = mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateCellMatrix(*choquetLayerCount,m_x_max);
    inv_sigma_out = plhs[3];
    
    //call computational routine
    choquetnode_core_mex(choquetLayerCount, CHOQUET_NODE_SIZE_TEMP, CHOQUET_INPUT_SIZE, hid_alpha,  V, G, x_diff_out, x_out, z_out, inv_sigma_out, m_x, m_x_max, m_y, n);
}