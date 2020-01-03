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

void x_init(double* x_new, double* x, double* layerCount, double* NETWORK_NODE_SIZE_TEMP, double* MLP_INPUT_SIZE, mwSize m, mwSize n)
{
    mwSize i,j,k,l,m_raw,ones_loc;

    m_raw=(unsigned int)*MLP_INPUT_SIZE;
    #pragma omp parallel for
    for (i=0;i<=m_raw;i=i+1)
    {
        for (j=0;j<n;j=j+1)
        {
            if (i==m_raw)
                x_new[(j*m)+i]=-1;
            else
                x_new[(j*m)+i]=x[(j*m_raw)+i];
        }
    }
    k=0;
    ones_loc=m_raw+NETWORK_NODE_SIZE_TEMP[k]+1;
    
    #pragma omp parallel for
    for (i=ones_loc;i<m;i=i+1)
    {
        if (i==ones_loc)
        {
            for (j=0;j<n;j=j+1)
            {
                x_new[(j*m)+i]=-1;
            }
            k=k+1;
            ones_loc=ones_loc+NETWORK_NODE_SIZE_TEMP[k]+1;
        }
    }
}

void perceptron_mex(double *y, double *y_diff, double *w, double *x, unsigned char activ_type, double alpha, mwSize m, mwSize n)
{
    mwSize i,j;
    long double v[n];
    
    for (i=0; i<n; i=i+1) {
        v[i]=0.0;
        for (j=0; j<m; j=j+1) {
            v[i] = v[i] + (long double)w[j] * (long double)x[(i*m)+j];
        }
    }

    for (i=0; i<n; i=i+1) {
        switch(activ_type) {
            case 0  :
                //elu
				if (v[i]>=0)
				{
					y[i]=(long double)v[i];
					y_diff[i]=1.0;
				}
				else
				{
					y[i]=(long double)alpha*(exp((long double)v[i])-1.0);
					y_diff[i]=(long double)y[i]+(long double)alpha;
				}

                break;
            
            case 1  :
                //tanh
                y[i]=(2.0/(1.0+exp(-(long double)alpha*(long double)v[i])))-1.0;
                y_diff[i]=(long double)alpha*((1.0-pow((long double)y[i],2.0))/2.0);
                break;

            case 2 :
                y[i]=v[i];
                y_diff[i]=1.0;
                
                break;
        }
    }
}

void multilayerperceptron_core_mex(double* layerCount, double* NETWORK_NODE_SIZE_TEMP, double* MLP_INPUT_SIZE, double hid_alpha, double out_alpha, const mxArray* w, double* x_diff_out, double* x_out, double* y_out, mwSize m_x, mwSize m_y, mwSize n)
{
    int i,j,k,l;
    mwSize m;
    
    unsigned int start_index, stop_index, cur_index;
    unsigned int NETWORK_NODE_SIZE_SUM_1, NETWORK_NODE_SIZE_SUM_2, NETWORK_NODE_SIZE_SUM_3;
    
    mxArray* w_temp;
    double* w_temp_array;
    
    double* x_temp;
    double* x_diff_out_temp;
    double* x_out_temp;
    x_out_temp=mxCalloc(n, sizeof(double));
    x_diff_out_temp=mxCalloc(n, sizeof(double));

    for (i=1; i<=(*layerCount); i=i+1)
    {
        NETWORK_NODE_SIZE_SUM_1=0; NETWORK_NODE_SIZE_SUM_2=0; 
        for (k=1; k<=(i-2); k=k+1)
        {
            NETWORK_NODE_SIZE_SUM_1=NETWORK_NODE_SIZE_SUM_1+NETWORK_NODE_SIZE_TEMP[k-1];
        }
        NETWORK_NODE_SIZE_SUM_2=NETWORK_NODE_SIZE_SUM_1;
        if ((i-1)>0)
        {
            NETWORK_NODE_SIZE_SUM_2=NETWORK_NODE_SIZE_SUM_2+NETWORK_NODE_SIZE_TEMP[i-2];
        }
        start_index=(*MLP_INPUT_SIZE+i-1)*unitstep(i-2)+NETWORK_NODE_SIZE_SUM_1;
        stop_index=(*MLP_INPUT_SIZE+i)+NETWORK_NODE_SIZE_SUM_2;
        m=stop_index-start_index;
        x_temp=mxCalloc(m*n, sizeof(double));
        
        #pragma omp parallel for
        for (j=0; j<n; j=j+1)
        {
			#pragma omp parallel for
            for (l=(start_index+1);l<=(stop_index);l=l+1)
            {
                x_temp[(j*m)+l-(start_index+1)]=x_out[(j*m_x)+l-1];
            }
        }
        
        for (k=1; k<=NETWORK_NODE_SIZE_TEMP[i-1]; k=k+1)
        {
            cur_index=NETWORK_NODE_SIZE_SUM_2+k-1;
            w_temp=mxGetCell(w, cur_index);
            w_temp_array=mxGetData(w_temp);
            
			if (i==(*layerCount))
                perceptron_mex(x_out_temp, x_diff_out_temp, w_temp_array, x_temp, 1, out_alpha, m, n);
			else
				perceptron_mex(x_out_temp, x_diff_out_temp, w_temp_array, x_temp, 0, hid_alpha, m, n);
            
			#pragma omp parallel for
            for (j=0; j<n; j=j+1)
            {
                x_out[(j*m_x)+(stop_index+k-1)]=x_out_temp[j];
                x_diff_out[(j*m_x)+(stop_index+k-1)]=x_diff_out_temp[j];
                if (i==(*layerCount))
                    y_out[(j*m_y)+k-1]=x_out_temp[j];
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
    
    mwSize i,m_x,m_y,n;
    //define variables
    double* layerCount;
    double* NETWORK_NODE_SIZE_TEMP;
    double hid_alpha,out_alpha;
    double* MLP_INPUT_SIZE;
    
    const mxArray* w;
	
    double* x_in;
	double* x_diff_out;
    double* x_out;
	double* y_out;
	
    //set input pointers
    layerCount = mxGetPr(prhs[0]);
    NETWORK_NODE_SIZE_TEMP = mxGetPr(prhs[1]);
    MLP_INPUT_SIZE = mxGetPr(prhs[2]);
    hid_alpha = mxGetScalar(prhs[3]);
    out_alpha = mxGetScalar(prhs[4]);
    w = prhs[5];
    x_in=mxGetPr(prhs[6]);
    
    m_x=mxGetM(prhs[6]);
    m_y=(unsigned int)(NETWORK_NODE_SIZE_TEMP[(unsigned int)(*layerCount-1)]);
    n=mxGetN(prhs[6]);

    //set output pointers
    plhs[0] = mxCreateDoubleMatrix(m_y,n,mxREAL);
    y_out = mxGetPr(plhs[0]);
    
    m_x=m_x+1;
    for (i=0;i<*layerCount;i=i+1)
    {
        m_x=m_x+NETWORK_NODE_SIZE_TEMP[i];
        if (i<(*layerCount-1))
        {
            m_x=m_x+1;
        }
    }
    plhs[1] = mxCreateDoubleMatrix(m_x,n,mxREAL);
    x_out=mxGetPr(plhs[1]);
    
    x_init(x_out, x_in, layerCount, NETWORK_NODE_SIZE_TEMP, MLP_INPUT_SIZE, m_x, n);

    plhs[2] = mxCreateDoubleMatrix(m_x,n,mxREAL);
    x_diff_out = mxGetPr(plhs[2]);

    //call computational routine
    multilayerperceptron_core_mex(layerCount, NETWORK_NODE_SIZE_TEMP, MLP_INPUT_SIZE, hid_alpha, out_alpha, w, x_diff_out, x_out, y_out, m_x, m_y, n);
}