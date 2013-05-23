#include <cstdio>
#include <iostream>
#include "bulk.hpp"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <cassert>

template<typename ThreadGroup, typename Iterator1, typename Size, typename Iterator2>
__device__
void block_copy_n(ThreadGroup &this_group, Iterator1 first, Size n, Iterator2 result)
{
  for(Size i = this_group.this_thread.index(); i < n; i += this_group.size())
  {
    result[i] = first[i];
  }

  this_group.wait();
}

struct reduce
{
  template<typename ThreadGroup>
  __device__
  void operator()(ThreadGroup &this_group, thrust::device_ptr<int> data, thrust::device_ptr<int> result)
  {
    unsigned int n = this_group.size();

    int *s_data = static_cast<int*>(bulk::malloc(this_group, n * sizeof(int)));

    block_copy_n(this_group, data, n, s_data);

    while(n > 1)
    {
      unsigned int half_n = n / 2;

      if(this_group.this_thread.index() < half_n)
      {
        s_data[this_group.this_thread.index()] += s_data[n - this_group.this_thread.index() - 1];
      }

      this_group.wait();

      n -= half_n;
    }

    this_group.wait();

    if(this_group.this_thread.index() == 0)
    {
      *result = s_data[0];
    }

    this_group.wait();

    bulk::free(this_group, s_data);
  }
};

int main()
{
  static const size_t group_size = 512;

  size_t n = group_size;

  thrust::device_vector<int> vec(n);

  thrust::sequence(vec.begin(), vec.end());

  thrust::device_vector<int> result(1);

  using bulk::par;
  using bulk::async;

  bulk::static_thread_group<group_size> group_spec;

  async(par(group_spec, 1), reduce(), bulk::there, vec.data(), result.data());

  assert(thrust::reduce(vec.begin(), vec.end()) == result[0]);
}

