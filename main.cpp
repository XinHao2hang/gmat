#include <iostream>
#include <ctime>
#include"gmat.h"
using namespace std;
#define COL 4096
#define ROW 4096
int main()
{
	INIT_GPU(0);
	float* n = new float[COL * ROW];
	float* r = new float[COL * ROW];
	for (int i = 0; i < ROW; i++)
	{
		for (int j = 0; j < COL; j++)
		{
			n[i * COL + j] = 2.0;
		}
	}
	gmat A,B;
	gmat D, C;
	C.setData(nullptr,COL,ROW, ALLOC_MEM_DEVICE);
	size_t localSize[2] = {8,8};
	gmat::setZero(C, localSize);
	//外来数据
	A.setData(n,ROW,COL, COPY_MEM_TO_DEVICE);
	B.setData(n, ROW, COL, COPY_MEM_TO_DEVICE);
	//delete[] n;
	cout << "start" << endl;
	time_t start, end;
	start = clock();
	//for(int i=0;i<100;i++)
	//{
	//	//C = A + B;
	//	gmat::add(A,B,C);
	//}
	gmat::mul(A,B,C);
	//C.print();
	end = clock();
	cout << "GPUok" <<" "<<end-start<<"ms"<< endl;
	end = 0;
	start = 0;
	start = clock();
	for (int k = 0; k < 100; k++)
	{
		for (int i = 0; i < COL; i++)
		{
			for (int j = 0; j < ROW; j++)
			{
				r[i * COL + j] = n[i * COL + j] + n[i * COL + j];
			}
		}
	}
	end = clock();
	cout << "CPUok" << " " << end - start << "ms" << endl;
	//cout << n[0] << endl;
	while (1);
	return 0;
}