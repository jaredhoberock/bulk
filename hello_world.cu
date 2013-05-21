#include <iostream>
#include <cstdio>
#include "bulk_async.hpp"

struct hello
{
  __host__ __device__
  void operator()(unsigned int num_threads)
  {
    unsigned int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(thread_idx < num_threads)
    {
      printf("Hello world!\n");
    }
  }
};

int main()
{
  bulk_async::bulk_async(bulk_async::launch(1), hello(), 1);

  cudaDeviceSynchronize();

  return 0;
}

