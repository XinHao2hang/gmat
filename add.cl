__kernel void main(__global float*pIn1,__global float*pIn2, __global float *pOut,__global int *row,__global int *col) 
{ 
   int i = get_global_id(0); 
   int j = get_global_id(1); 
   int index = i*(*col)+j;
   pOut[index]=pIn1[index]+pIn2[index];

} 
