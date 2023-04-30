#include "gtest.h"

__global__ void g_add(int n, float* x, float* y)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;
	//for (int i = index; i < 1 ; i += stride)
	y[idx] += x[idx];
}

__global__ void g_setup(float* x, float* y)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	x[idx] = 1.0f;
	y[idx] = 2.0f;
}

int g_main()
{
	int n = 1 << 20;
	float* x;
	float* y;
	cudaMallocManaged(&x, n * sizeof(float));
	cudaMallocManaged(&y, n * sizeof(float));

	int tpb = 256;
	int bpg = (n + tpb - 1) / tpb;

	g_setup<<<bpg, tpb>>>(x, y);

	auto start = std::chrono::high_resolution_clock::now();
	g_add<<<bpg, tpb>>>(n, x, y);
	cudaDeviceSynchronize();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << duration.count() << std::endl;

	float maxError = 0.0f;
	for (int i = 0; i < n; i++)
	{
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max Error : " << maxError << std::endl;

	cudaFree(x);
	cudaFree(y);
	return 690;
}