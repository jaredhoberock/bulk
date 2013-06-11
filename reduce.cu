#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cassert>
#include <iostream>

struct reduce_partitions
{
  template<typename ExecutionGroup, typename Iterator1, typename Size, typename Iterator2, typename BinaryFunction>
  __device__
  void operator()(ExecutionGroup &this_group, Iterator1 first, Iterator1 last, Size partition_size, Iterator2 result, BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<Iterator1>::type value_type;

    Iterator1 partition_first = first + partition_size * this_group.index();

    Iterator1 partition_last = thrust::min(partition_first + partition_size, last);

    value_type sum = bulk::reduce(this_group, partition_first, partition_last, value_type(0), binary_op);

    if(this_group.this_exec.index() == 0)
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
T my_reduce(Iterator first, Iterator last, T init, BinaryOperation binary_op)
{
  int n = last - first;

  if(n <= 0) return init;

  unsigned int num_processors = 16;

  unsigned int subscription = 4;

  static const size_t group_size = 512;

  const unsigned int partition_size = thrust::max<unsigned int>(group_size, divide_ri(n, subscription * num_processors));

  unsigned int num_partial_sums = thrust::max<unsigned int>(1, divide_ri(n, partition_size));

  thrust::device_vector<T> partial_sums(num_partial_sums);

  bulk::static_execution_group<group_size> g;

  bulk::async(bulk::par(g, partial_sums.size()), reduce_partitions(), bulk::there, first, last, partition_size, partial_sums.begin(), binary_op);

  // we only need a single additional step because partition_size > subscription * num_processors
  if(partial_sums.size() > 1)
  {
    bulk::async(bulk::par(g, 1), reduce_partitions(), bulk::there, partial_sums.begin(), partial_sums.end(), partial_sums.size(), partial_sums.begin(), binary_op);
  } // end while

  return partial_sums[0];
}


int main()
{
  size_t n = 123456789;

  thrust::device_vector<int> vec(n);

  thrust::sequence(vec.begin(), vec.end());

  int my_result = my_reduce(vec.begin(), vec.end(), 0, thrust::plus<int>());

  std::cout << "my_result: " << my_result << std::endl;

  int thrust_result = thrust::reduce(vec.begin(), vec.end(), 0, thrust::plus<int>());

  std::cout << "thrust_result: " << thrust_result << std::endl;

  assert(thrust_result == my_result);
}

