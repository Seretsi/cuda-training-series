
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

using namespace std;

void matFill(vector<vector<float>>& mat, const size_t size);
void matReserve(vector<vector<float>>& mat, const size_t size);

int main()
{
    size_t size = 3;
    // init host variables
    vector<vector<float>> a, b, c;
    matFill(a, size);
    matFill(b, size);
    matReserve(c, size);
    // init device vars
    int* dev_a, dev_b, dev_c;
    cudaMalloc((void**)&dev_a, size * size * sizeof(float));
    cudaMalloc((void**)&dev_b, size * size * sizeof(float));
    cudaMalloc((void**)&dev_c, size * size * sizeof(float));
    // alloc device mem

    // transfer mem to device

    // start kernel

    // wait for conclusion

    // print result

    // clean up host and device memory

    return 0;
}

void matFill(vector<vector<int>>& mat, const size_t size)
{
    mat.reserve(size);
    for (auto m : mat)
    {
        m.reserve(size);
        for (auto n : m)
            n = rand() % 10;
    }
}

void matReserve(vector<vector<int>>& mat, const size_t size)
{
    mat.reserve(size);
    for (auto m : mat)
    {
        m.reserve(size);
    }
}