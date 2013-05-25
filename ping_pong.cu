#include <thrust/device_vector.h>
#include <bulk/bulk.hpp>
#include <cstdio>

struct ping
{
  __device__
  void operator()(volatile int *ball)
  {
    if(threadIdx.x == 0)
    {
      *ball = 1;

      for(unsigned int next_state = 2;
          next_state < 25;
          next_state += 2)
      {
        while(*ball != next_state)
        {
          printf("ping waiting for return\n");
        }

        *ball += 1;

        printf("ping! ball is now %d\n", next_state + 1);
      }
    }
  }
};

struct pong
{
  __device__
  void operator()(volatile int *ball)
  {
    if(threadIdx.x == 0)
    {
      for(unsigned int next_state = 1;
          next_state < 25;
          next_state += 2)
      {
        while(*ball != next_state)
        {
          printf("pong waiting for return\n");
        }

        *ball += 1;

        printf("pong! ball is now %d\n", next_state + 1);
      }
    }
  }
};

int main()
{
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  using bulk::par_async;
  using bulk::async;

  thrust::device_vector<int> ball(1);

  bulk::future<void> t1 = async(par_async(s1, 1), ping(), thrust::raw_pointer_cast(&*ball.data()));
  bulk::future<void> t2 = async(par_async(s2, 1), pong(), thrust::raw_pointer_cast(&*ball.data()));

  t1.wait();
  t2.wait();

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  return 0;
}

