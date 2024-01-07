
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

using namespace std;

template <typename T>
void matFill(vector<T>& mat, const size_t size);
template <typename T>
void printMat(vector<T> mat, size_t t);

const int BLOCK_SIZE = 32;

__global__ void matmul_sharedMemory(const float* a, const float* b, float* c, const size_t size)
{
    __shared__ int localMemorya[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int localMemoryb[BLOCK_SIZE][BLOCK_SIZE];

    // load shared 
    int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    localMemorya[idxx][idxy] = a[idxx];
    // wait for completion

    // process matmul
}

int main()
{
    const size_t size = 3;
    // init host variables
    vector<float> a, b, c;
    matFill(a, size);
    matFill(b, size);
    // init device vars
    float *dev_a, *dev_b, *dev_c;
    // alloc device mem
    cudaMalloc((void**)&dev_a, size * size * sizeof(float));
    cudaMalloc((void**)&dev_b, size * size * sizeof(float));
    cudaMalloc((void**)&dev_c, size * size * sizeof(float));
    // transfer mem to device
    cudaMemcpy(&dev_a, a.data(), size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_b, b.data(), size * size * sizeof(float), cudaMemcpyHostToDevice);
    // start kernel
    matmul_sharedMemory <<< size, size >>> (dev_a, dev_b, dev_c, size);
    c.reserve(size);
    // wait for conclusion
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), &dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    // print result
    printMat(c, size);
    // clean up host and device memory
    cudaFree(&dev_a);
    cudaFree(&dev_b);
    cudaFree(&dev_c);
    return 0;
}

template <typename T>
void matFill(vector<T>& mat, const size_t size)
{
    mat.reserve(size);
    for (auto n : mat)
    {
       n = rand() % 10;
    }
}

template<typename T>
void printMat(vector<T> mat, size_t t)
{
    size_t ctr = 1;
    for (auto n : mat)
    {
        printf("%d, ", n);
        if (ctr%t == 0)
            printf("|/n");
    }
}
