__kernel void main(__global float*mat) 
{ 
   int i = get_global_id(0); 
   int j = get_global_id(1); 
   mat[i*(*col)+j]=0;
} 
