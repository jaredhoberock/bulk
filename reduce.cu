#include <cstdio>
#include <iostream>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cassert>

struct reduce_partitions
{
  template<typename ThreadGroup, typename Iterator1, typename Iterator2>
  __device__
  void operator()(ThreadGroup &this_group, Iterator1 first, Iterator1 last, unsigned int partition_size, Iterator2 result)
  {
    Iterator1 partition_first = first + partition_size * this_group.index();

    Iterator1 partition_last = thrust::min(first + partition_size, last);

    int sum = bulk::reduce(this_group, partition_first, partition_last, 0, thrust::plus<int>());

    if(this_group.this_thread.index() == 0)
    {
      result[this_group.index()] = sum;
    }
  }
};


int divide_ri(int n, int d)
{
  return (n + (d - 1)) / d;
}


template<typename Iterator,
         typename T,
         typename BinaryOperation>
T reduce(Iterator first, Iterator last, T init, BinaryOperation binary_op)
{
  int n = last - first;

  if(n <= 0) return init;

  typedef typename thrust::iterator_value<Iterator>::type value_type;

  unsigned int num_processors = 16;

  unsigned int subscription = 4;

  static const size_t group_size = 512;

  unsigned int num_partial_sums = thrust::min<unsigned int>(subscription * num_processors, n);

  thrust::device_vector<value_type> partial_sums(num_partial_sums);

  bulk::static_thread_group<group_size> g;

  const unsigned int partition_size = thrust::max<unsigned int>(group_size, divide_ri(n, partial_sums.size()));

  bulk::async(bulk::par(g, partial_sums.size()), reduce_partitions(), bulk::there, first, last, partition_size, partial_sums.begin());

  // we only need a single additional step because partition_size > subscription * num_processors
  if(partial_sums.size() > 1)
  {
    bulk::async(bulk::par(g, partial_sums.size()), reduce_partitions(), bulk::there, partial_sums.begin(), partial_sums.end(), partial_sums.size(), partial_sums.begin());
  } // end while

  return partial_sums[0];
}


int main()
{
  static const size_t group_size = 123456789;

  size_t n = group_size;

  thrust::device_vector<int> vec(n);

  thrust::sequence(vec.begin(), vec.end());

  int result = ::reduce(vec.begin(), vec.end(), 0, thrust::plus<int>());

  assert(thrust::reduce(vec.begin(), vec.end()) == result);
}

