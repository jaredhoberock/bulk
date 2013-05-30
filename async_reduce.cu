#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <bulk/bulk.hpp>
#include <cassert>


struct reduce_kernel
{
  template<typename Iterator, typename Pointer>
  __device__ void operator()(volatile bool wait_for_me, Iterator first, Iterator last, Pointer result)
  {
    while(!wait_for_me){}

    *result = thrust::reduce(thrust::device, first, last);
  }
};


struct greenlight
{
  __device__ void operator()(bool *set_me)
  {
    *set_me = true;
  }
};


int main()
{
  cudaStream_t s1,s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  using bulk::par_async;
  using bulk::async;

  thrust::device_vector<int> vec(1 << 20);
  thrust::sequence(vec.begin(), vec.end());

  thrust::device_vector<int> result(1);
  thrust::device_vector<bool> flag(1);

  // note we launch the reduction before the greenlight
  async(par_async(s1,1), reduce_kernel(), thrust::raw_pointer_cast(flag.data()), vec.begin(), vec.end(), result.begin());

  async(par_async(s2,1), greenlight(), thrust::raw_pointer_cast(flag.data()));

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  assert(thrust::reduce(vec.begin(), vec.end()) == result[0]);

  std::cout << "result: " << thrust::reduce(vec.begin(), vec.end()) << std::endl;
  std::cout << "asynchronous result: " << result[0] << std::endl;

  return 0;
}

