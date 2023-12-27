#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>

__global__ void matmul(float* a, float* b, float* c, const int X)
{	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < X * X)
	{
		float result = 0;
		c[index] = 0;
		for (int i = 0; i < X; i++)
		{
			int idxA = i + blockDim.x * blockIdx.x;
			int idxB = threadIdx.x + blockDim.x * i;
			c[index] += a[idxA] * b[idxB];
		}
	}
}

template <typename T>
__host__ void renderMat(std::vector<T>& mat);

int main()
{
	// init variables
	std::vector<float> a, b, c;
	const int X = 3;
	a.resize(X * X);
	b.resize(X * X);
	for (int i = 0; i < X * X; i++)
	{
		a[i] = rand()%3;
		b[i] = rand()%3;
	}
	float *dev_a, *dev_b, *dev_c;
	auto size = X * X * sizeof(float);
	// allocate memory
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	// copy memory
	cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice);
	// run kernel
	matmul<<<X, X>>>(dev_a, dev_b, dev_c, X);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		// Handle the error appropriately
	}
	c.resize(X * X);
	cudaDeviceSynchronize();
	// copy result
	cudaMemcpy(c.data(), dev_c, size, cudaMemcpyDeviceToHost);
	// render result
	renderMat(a);
	printf("=================\n");
	renderMat(b);
	printf("=================\n");
	renderMat(c);
	// free memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}

template <typename T>
__host__ void renderMat(std::vector<T>& mat)
{
	size_t size = sqrt(mat.size());
	for (size_t i = 0; i <mat.size(); i++)
	{
		printf("%f | ", mat[i]);
		if ((i + 1) % size == 0)
			printf("\n");
	}
}
