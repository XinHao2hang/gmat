__kernel void main(__global float*pIn1,__global float*pIn2, __global float *pOut,__global int *row,__global int *col,__global int *num) 
{
   int i = get_global_id(0); 
   int j = get_global_id(1);
   pOut[i*(*col)+j]+=pIn1[*num*(*col)+j]*pIn2[i*(*col)+*num];
} 
