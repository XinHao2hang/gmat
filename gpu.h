#pragma once
#include <cl.h>
#include <vector>
class gpu;
class gpu_program
{
private:
	//��ȡ����
	char* readCode(const char* filename);
	cl_program createProgram(const char* code, cl_context context, cl_device_id* device, int dev_id);
	//������ִ�д���
	void createCode(const char* filename, const char* entry, gpu& _gpu);
public:
	//����
	cl_kernel kernel;
	//����(ȫ����ָ�����ʽ����)
	std::vector<cl_mem> data;
	void loadProgram(const char* filename, const char* entry, gpu& _gpu);
	//�������
	void setArgs(std::vector<void*> _data) 
	{ 
		for(int i=0;i<_data.size();i++)
			clSetKernelArg(kernel, i, sizeof(void*), _data[i]);
	}
	//��������
	void clearArgs() { data.clear(); }
	gpu_program();
	~gpu_program();
};
class gpu
{
public:
	//���м����ڲ����
	cl_device_id* devices;//�豸
	cl_context _context;//������
	cl_command_queue _commandQueue;//�������
	int dev_id=0;
	int argNum = 0;

	gpu();
	//��ʼ��OpenCL
	cl_int CL_Init(int plat_id);
	//�ڴ����루����ָ��,��С��
	cl_mem gmalloc(void* data, unsigned long size, cl_mem_flags flags);
	//����
	void run(int dim, gpu_program program, size_t size_xy[2], size_t localWorkSize[2]);

	~gpu();
};