#pragma once
#include"gpu.h"
#define INIT_GPU(x) gmat::gpu_config(x)
#define ADD 0
#define SUB 3
#define MUL 1
#define ZERO 2
#define ALLOC_MEM_HOST CL_MEM_ALLOC_HOST_PTR
#define ALLOC_MEM_DEVICE CL_MEM_READ_WRITE
#define USE_MEM_HOST CL_MEM_USE_HOST_PTR
#define COPY_MEM_TO_DEVICE CL_MEM_COPY_HOST_PTR
//这个需要new的方式来分配
struct DataPack
{
	//数据
	cl_mem data;
	//计数器
	unsigned long count;
	//标记加一
	void occupy(){ ++count; }
	bool release()
	{ 
		--count; 
		if (count <= 0)
		{
			clReleaseMemObject(data);
			return true;
		}
		else
		{
			return false;
		}
	}
};
struct Size
{
	unsigned long row;
	unsigned long col;
};
class gmat
{
	//设备和GPU代码
	static gpu GPU;
	static std::vector<gpu_program> programs;
public:
	int row, col;
	DataPack* data_pack = nullptr;
	gmat();
	gmat(const gmat& mat);
	static void gpu_config(int gpu_id);
	void setData(float* _data,unsigned long row,unsigned long col, cl_mem_flags flags);
	void operator=(gmat mat);
	gmat operator+(gmat mat);
	static void mul(gmat&A,gmat&B,gmat&C);
	static void add(gmat&A,gmat&B,gmat&C);
	static void setZero(gmat& mat, size_t localWorkSize[2]);
	void print();
	~gmat();
};

