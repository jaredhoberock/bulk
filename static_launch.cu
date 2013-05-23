#include <cstdio>
#include <iostream>
#include "bulk_async.hpp"
#include "shmalloc.hpp"
#include "thread_group.hpp"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <cassert>

template<typename ThreadGroup, typename Iterator1, typename Size, typename Iterator2>
__device__
void block_copy_n(ThreadGroup &this_group, Iterator1 first, Size n, Iterator2 result)
{
  for(Size i = this_group.index(); i < n; i += this_group.size())
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
    __shared__ int *s_s_data;

    unsigned int n = this_group.size();

    if(this_group.index() == 0)
    {
      s_s_data = static_cast<int *>(bulk_async::shmalloc(n * sizeof(int)));
    }
    this_group.wait();

    int *s_data = s_s_data;

    block_copy_n(this_group, data, n, s_data);

    while(n > 1)
    {
      unsigned int half_n = n / 2;

      if(this_group.index() < half_n)
      {
        s_data[this_group.index()] += s_data[n - this_group.index() - 1];
      }

      this_group.wait();

      n -= half_n;
    }

    this_group.wait();

    if(this_group.index() == 0)
    {
      *result = s_data[0];
      bulk_async::shfree(s_data);
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

  using bulk_async::launch;

  bulk_async::static_thread_group<group_size> group_spec;

  // size smem ourself
  bulk_async::bulk_async(launch(group_spec, 1, group_size * sizeof(int)), reduce(), bulk_async::there, vec.data(), result.data());

  assert(thrust::reduce(vec.begin(), vec.end()) == result[0]);
}

