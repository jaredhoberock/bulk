#include <cstdio>
#include <iostream>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <cassert>

struct reduce_kernel
{
  template<typename ThreadGroup>
  __device__
  void operator()(ThreadGroup &this_group, thrust::device_ptr<int> data, thrust::device_ptr<int> result)
  {
    unsigned int n = this_group.size();

    int sum = bulk::reduce(this_group, data, data + n, 0, thrust::plus<int>());

    if(this_group.this_thread.index() == 0)
    {
      *result = sum;
    }
  }
};

int main()
{
  static const size_t group_size = 512;

  size_t n = group_size;

  thrust::device_vector<int> vec(n);

  thrust::sequence(vec.begin(), vec.end());

  thrust::device_vector<int> result(1);

  bulk::static_thread_group<group_size> group_spec;

  bulk::async(bulk::par(group_spec, 1), reduce_kernel(), bulk::there, vec.data(), result.data());

  assert(thrust::reduce(vec.begin(), vec.end()) == result[0]);
}

