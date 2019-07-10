#include "gmat.h"
#include<iostream>
using namespace std;
gpu gmat::GPU = {};
std::vector<gpu_program> gmat::programs;
const char* kenrnelFiles[16] = {"add.cl","mul.cl","zero.cl"};//文件名
gmat::gmat()
{
	data_pack = new DataPack;
	data_pack->count = 0;
}

gmat::gmat(const gmat& mat)
{
	memcpy(this, &mat, sizeof(gmat));
	data_pack->occupy();
}

void gmat::gpu_config(int gpu_id)
{
	GPU.CL_Init(gpu_id);
	//加载程序
	for (int i = 0; i < 3; i++)
	{
		gpu_program program;
		program.loadProgram(kenrnelFiles[i], "main", GPU);
		programs.push_back(program);
	}
}

void gmat::setData(float* _data, unsigned long _row, unsigned long _col, cl_mem_flags flags)
{
	//这里申请显存
	if (data_pack->count == 0)
	{
		row = _row;
		col = _col;
		data_pack->data = GPU.gmalloc(_data, _row*_col*sizeof(float), flags);
		data_pack->count = 1;
	}
}

void gmat::operator=(gmat mat)
{
	mat.data_pack->occupy();
	//释放原始数据
	if (data_pack != nullptr)
		data_pack->release();
	//浅层复制数据
	memcpy(this,&mat,sizeof(gmat));
}

gmat gmat::operator+(gmat mat)
{
	gmat res;
	res.setData(nullptr,row,col, ALLOC_MEM_HOST);
	//外来数据
	cl_mem t_row = GPU.gmalloc(&row,4, COPY_MEM_TO_DEVICE);
	cl_mem t_col = GPU.gmalloc(&col, 4, COPY_MEM_TO_DEVICE);
	//输入参数
	programs[ADD].setArgs({&(data_pack->data),&(mat.data_pack->data),&(res.data_pack->data),&t_row,&t_col});
	//执行代码
	size_t localWorkSize[2] = { 1,1 };
	size_t size_xy[2] = { row,col };
	GPU.run(2, programs[ADD],size_xy ,localWorkSize );
	//清除参数
	programs[ADD].clearArgs();
	return res;
}

void gmat::add(gmat& A, gmat& B, gmat& C)
{
	//外来数据
	cl_mem t_row = GPU.gmalloc(&A.row, 4, COPY_MEM_TO_DEVICE);
	cl_mem t_col = GPU.gmalloc(&A.col, 4, COPY_MEM_TO_DEVICE);
	//输入参数
	programs[ADD].setArgs({ &(A.data_pack->data),&(B.data_pack->data),&(C.data_pack->data),&t_row,&t_col });
	//执行代码
	size_t localWorkSize[2] = { 8,8 };
	size_t size_xy[2] = { A.row,A.col };
	GPU.run(2, programs[ADD], size_xy, localWorkSize);
	//清除参数
	programs[ADD].clearArgs();
	clReleaseMemObject(t_row);
	clReleaseMemObject(t_col);

}

void gmat::setZero(gmat& mat,size_t localWorkSize[2])
{
	//输入参数
	programs[ZERO].setArgs({&mat.data_pack->data});
	size_t size_xy[2] = { mat.row,mat.col };
	GPU.run(2, programs[ADD], size_xy, localWorkSize);
	programs[ZERO].clearArgs();
}

void gmat::mul(gmat& A, gmat& B, gmat& C)
{
	int i = 0;
	cl_mem num = GPU.gmalloc(&i, 4, USE_MEM_HOST);
	cl_mem t_row = GPU.gmalloc(&A.row, 4, USE_MEM_HOST);
	cl_mem t_col = GPU.gmalloc(&A.col, 4, USE_MEM_HOST);
	programs[MUL].setArgs({ &(A.data_pack->data),&(B.data_pack->data),&(C.data_pack->data),&t_row,&t_col ,&num });
	size_t localWorkSize[2] = { 8,8 };
	size_t size_xy[2] = { A.row,B.col };
	for (i = 0; i < A.col; i++)
	{
		GPU.run(2, programs[MUL], size_xy, localWorkSize);
	}
	//清除参数
	programs[MUL].clearArgs();
	clReleaseMemObject(num);
	clReleaseMemObject(t_row);
	clReleaseMemObject(t_col);
}


void gmat::print()
{
	float* out = new float[row * col];
	clEnqueueReadBuffer(GPU._commandQueue, data_pack->data, CL_TRUE, 0,row*col*sizeof(float), out, 0, NULL, NULL);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			cout << out[i * row + j]<<" ";
		cout << endl;
	}
}

gmat::~gmat()
{
	//释放占用的空间
	if (data_pack->release())
		delete[] data_pack;
}

