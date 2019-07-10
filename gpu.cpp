#include "gpu.h"
#include <memory.h>

char* gpu_program::readCode(const char* filename)
{
	FILE* fp;
	fp = fopen(filename, "rb+");
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* code = new char[len + 2];
	fread(code, 1, len, fp);
	code[len] = 0;
	fclose(fp);
	return code;
}

cl_program gpu_program::createProgram(const char* code, cl_context context, cl_device_id* device, int dev_id)
{
	cl_program program = NULL;
	size_t len = strlen(code);
	program = clCreateProgramWithSource(context, 1, &code, &len, NULL);
	if (program == NULL)
		return program;
	if (clBuildProgram(program, 1, device, NULL, NULL, NULL) != CL_SUCCESS)
	{
		char szBuildLog[4096];
		clGetProgramBuildInfo(program, device[dev_id], CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);
		//cout << "Error in Kernel: " << endl << szBuildLog;
		return NULL;
	}
	return program;
}

void gpu_program::createCode(const char* filename, const char* entry, gpu& _gpu)
{
	//创建代码
	char* code = readCode(filename);
	//创建着色器代码
	cl_program program = createProgram(code, _gpu._context, _gpu.devices, _gpu.dev_id);
	//创建内核空间
	kernel = clCreateKernel(program, entry, NULL);
	//删除空间
	delete[] code;
}

void gpu_program::loadProgram(const char* filename, const char* entry, gpu& _gpu)
{
	//创建代码
	char* code = readCode(filename);
	//创建着色器代码
	cl_program program = createProgram(code, _gpu._context, _gpu.devices, _gpu.dev_id);
	//创建内核空间
	kernel = clCreateKernel(program, entry, NULL);
	//删除空间
	delete[] code;
}

gpu_program::gpu_program()
{
}


gpu_program::~gpu_program()
{
}



gpu::gpu()
{
}

cl_int gpu::CL_Init(int plat_id)
{
	cl_uint platformNum = 0;
	//获取平台数目,intel,amd,nvidia....
	cl_int status = clGetPlatformIDs(0, NULL, &platformNum);
	if (status != CL_SUCCESS)
		return 1;

	//获取指定平台ID
	cl_platform_id	platform = NULL;
	if (platformNum > 0)
	{
		cl_platform_id* p_id = new cl_platform_id[platformNum];
		memset(p_id, 0, platformNum * sizeof(cl_platform_id));
		//获得平台ID
		clGetPlatformIDs(platformNum, p_id, NULL);
		platform = p_id[plat_id];
		//delete[] p_id;
	}
	if (platform == NULL)
		return 2;

	//获取对应平台GPU数目
	cl_uint deviceNum = 0;
	cl_device_id * device = NULL;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNum);
	if (deviceNum == 0)
	{
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &deviceNum);
		// 为设备分配空间
		device = new cl_device_id[deviceNum];// (cl_device_id*)malloc(deviceNum * sizeof(cl_device_id));
		// 获得平台
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, deviceNum, device, NULL);
	}
	else
	{
		device = new cl_device_id[deviceNum];
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceNum, device, NULL);
	}

	//创建设备上下文
	cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, NULL);
	if (NULL == context)
		return 3;

	//创建命令队列
	cl_command_queue commandQueue = clCreateCommandQueue(context, device[0], 0, NULL);
	if (NULL == commandQueue)
		return 4;
	//返回重要信息
	devices = device;
	_context = context;
	_commandQueue = commandQueue;
	return 0;
}
cl_mem gpu::gmalloc(void* data, unsigned long size, cl_mem_flags flags)
{
	return clCreateBuffer(_context, flags, size, data, NULL);
}

void gpu::run(int dim, gpu_program program, size_t size_xy[2], size_t localWorkSize[2])
{
	//计算
	clEnqueueNDRangeKernel(_commandQueue, program.kernel, dim, NULL, size_xy, localWorkSize, 0, NULL, NULL);
}
gpu::~gpu()
{
}
