#pragma once
#include <cl.h>
#include <vector>
class gpu;
class gpu_program
{
private:
	//读取代码
	char* readCode(const char* filename);
	cl_program createProgram(const char* code, cl_context context, cl_device_id* device, int dev_id);
	//创建可执行代码
	void createCode(const char* filename, const char* entry, gpu& _gpu);
public:
	//程序
	cl_kernel kernel;
	//参数(全部以指针的形式传入)
	std::vector<cl_mem> data;
	void loadProgram(const char* filename, const char* entry, gpu& _gpu);
	//输入参数
	void setArgs(std::vector<void*> _data) 
	{ 
		for(int i=0;i<_data.size();i++)
			clSetKernelArg(kernel, i, sizeof(void*), _data[i]);
	}
	//撤销参数
	void clearArgs() { data.clear(); }
	gpu_program();
	~gpu_program();
};
class gpu
{
public:
	//并行计算内部句柄
	cl_device_id* devices;//设备
	cl_context _context;//上下文
	cl_command_queue _commandQueue;//命令队列
	int dev_id=0;
	int argNum = 0;

	gpu();
	//初始化OpenCL
	cl_int CL_Init(int plat_id);
	//内存申请（数据指针,大小）
	cl_mem gmalloc(void* data, unsigned long size, cl_mem_flags flags);
	//运行
	void run(int dim, gpu_program program, size_t size_xy[2], size_t localWorkSize[2]);

	~gpu();
};