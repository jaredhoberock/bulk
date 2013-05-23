#include <iostream>
#include <cstdio>
#include "bulk.hpp"

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
  bulk::async(bulk::launch(1), hello(), 1);

  cudaDeviceSynchronize();

  return 0;
}

