#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
__global__ void matmul(int *c, const int *a, const int *b, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		c[i] = a[i] + b[i];
}

int main()
{
	const int n = 100000;
	std::vector<int> a(n), b(n), c(n);
	int *dev_a, *dev_b, *dev_c;

	for (int i = 0; i < n; ++i)
	{
		a[i] = rand()%10;
		b[i] = rand()%10;
	}
	// allocate in device memory
	cudaMalloc((void**)&dev_a, n * sizeof(int));
	cudaMalloc((void**)&dev_b, n * sizeof(int));
	cudaMalloc((void**)&dev_c, n * sizeof(int));
	// copy to device memory
	cudaMemcpy(dev_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	// execute kernel
	matmul<<< (n + 1024+1)/1024, 1024 >>>(dev_c, dev_a, dev_b, n);
	c.resize(n);
	cudaDeviceSynchronize();
	// copy back to host memory
	cudaMemcpy(c.data(), dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

	// print results
	for (int i = 0; i < 10; ++i)
	{
		printf("%d: %d + %d = %d\n", i, a[i], b[i], c[i]);
	}
	// free device memory
	cudaFree(&dev_a);
	cudaFree(&dev_b);
	cudaFree(&dev_c);
}