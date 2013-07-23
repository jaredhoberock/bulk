#include <cstdio>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <cassert>

template<typename Iterator1, typename Size, typename Iterator2>
__device__
void block_copy_n(Iterator1 first, Size n, Iterator2 result)
{
  for(Size i = threadIdx.x; i < n; i += blockDim.x)
  {
    result[i] = first[i];
  }

  __syncthreads();
}

struct sum
{
  __device__
  void operator()(thrust::device_ptr<int> data, thrust::device_ptr<int> result)
  {
    __shared__ int *s_s_data;

    unsigned int n = blockDim.x;

    if(threadIdx.x == 0)
    {
      s_s_data = static_cast<int *>(bulk::shmalloc(n * sizeof(int)));
    }
    __syncthreads();

    int *s_data = s_s_data;

    block_copy_n(data, n, s_data);

    while(n > 1)
    {
      unsigned int half_n = n / 2;

      if(threadIdx.x < half_n)
      {
        s_data[threadIdx.x] += s_data[n - threadIdx.x - 1];
      }

      __syncthreads();

      n -= half_n;
    }

    __syncthreads();
    if(threadIdx.x == 0)
    {
      *result = s_data[0];
      bulk::shfree(s_data);
    }
  }
};

int main()
{
  size_t block_size = 512;

  size_t n = block_size;

  thrust::device_vector<int> vec(n);

  thrust::sequence(vec.begin(), vec.end());

  thrust::device_vector<int> result(1);

  using bulk::par;
  using bulk::con;

  // let the runtime size smem
  bulk::async(par(con(block_size), 1), sum(), vec.data(), result.data());

  assert(thrust::reduce(vec.begin(), vec.end()) == result[0]);

  // size smem ourself
  size_t heap_size = block_size * sizeof(int);
  bulk::async(par(con(block_size, heap_size), 1), sum(), vec.data(), result.data());

  assert(thrust::reduce(vec.begin(), vec.end()) == result[0]);
}

