#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cassert>
#include <iostream>
#include "time_invocation_cuda.hpp"

// compute the indices of the first and last (exclusive) partitions the group will consume
template<typename Size>
__device__
thrust::pair<Size,Size> tile_range_in_partitions(Size tile_index,
                                                 Size num_partitions_per_tile,
                                                 Size last_partial_partition_size)
{
  thrust::pair<Size,Size> result;

  result.first = num_partitions_per_tile * tile_index;
  result.first += thrust::min<Size>(tile_index, last_partial_partition_size);

  result.second = result.first + num_partitions_per_tile + (tile_index < last_partial_partition_size);

  return result;
} // end tile_range_in_partitions()


// compute the indices of the first and last (exclusive) elements the group will consume
template<typename Size>
__device__
thrust::pair<Size,Size> tile_range(Size tile_index,
                                   Size num_partitions_per_tile,
                                   Size last_partial_partition_size,
                                   Size partition_size,
                                   Size n)
{
  thrust::pair<Size,Size> result = tile_range_in_partitions(tile_index, num_partitions_per_tile, last_partial_partition_size);
  result.first *= partition_size;
  result.second = thrust::min<Size>(n, result.second * partition_size);
  return result;
} // end tile_range()


struct reduce_partitions
{
  template<typename ExecutionGroup, typename Iterator1, typename Iterator2, typename T, typename BinaryOperation>
  __device__
  void operator()(ExecutionGroup &this_group, Iterator1 first, Iterator1 last, Iterator2 result, T init, BinaryOperation binary_op)
  {
    T sum = bulk::reduce(this_group, first, last, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      *result = sum;
    }
  }

  template<typename ExecutionGroup, typename Iterator1, typename Iterator2, typename BinaryOperation>
  __device__
  void operator()(ExecutionGroup &this_group, Iterator1 first, Iterator1 last, Iterator2 result, BinaryOperation binary_op)
  {
    // noticeably faster to pass the last element as the init
    (*this)(this_group, first, last - 1, result, thrust::raw_reference_cast(last[-1]), binary_op);
  }


  template<typename ExecutionGroup, typename Iterator1, typename Size, typename Iterator2, typename T, typename BinaryFunction>
  __device__
  void operator()(ExecutionGroup &this_group, Iterator1 first, Iterator1 last, Size num_partitions_per_tile, Size partition_size, Size last_partial_partition_size, Iterator2 result, T init, BinaryFunction binary_op)
  {
    thrust::pair<Size,Size> range = tile_range<Size>(this_group.index(), num_partitions_per_tile, last_partial_partition_size, partition_size, last - first);

    last = first + range.second;
    first += range.first;

    if(this_group.index() != 0)
    {
      // noticeably faster to pass the last element as the init 
      init = last[-1];
      --last;
    } // end if

    (*this)(this_group, first, last, result + this_group.index(), init, binary_op);
  }
};


template<typename RandomAccessIterator,
         typename T,
         typename BinaryOperation>
T my_reduce(RandomAccessIterator first, RandomAccessIterator last, T init, BinaryOperation binary_op)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type size_type;

  const size_type n = last - first;

  if(n <= 0) return init;

  const size_type subscription = 10;
  bulk::static_execution_group<128,9> g;

  const size_type partition_size = g.size() * g.grainsize();
  const size_type num_partitions = (n + partition_size - 1) / partition_size;
  const size_type num_tiles = thrust::min<size_type>(num_partitions, subscription * g.hardware_concurrency());
  const size_type num_partitions_per_tile = num_partitions / num_tiles;
  const size_type last_partial_partition_size = num_partitions % num_tiles;

  thrust::cuda::tag t;
  thrust::detail::temporary_array<T,thrust::cuda::tag> partial_sums(t, num_tiles);

  // reduce into partial sums
  bulk::async(bulk::par(g, partial_sums.size()), reduce_partitions(), bulk::there, first, last, num_partitions_per_tile, partition_size, last_partial_partition_size, partial_sums.begin(), init, binary_op);

  if(partial_sums.size() > 1)
  {
    // reduce the partial sums
    bulk::async(bulk::par(g, 1), reduce_partitions(), bulk::there, partial_sums.begin(), partial_sums.end(), partial_sums.begin(), binary_op);
  } // end while

  return partial_sums[0];
} // end my_reduce()


template<typename T>
T my_reduce(const thrust::device_vector<T> *vec)
{
  return my_reduce(vec->begin(), vec->end(), T(0), thrust::plus<T>());
}


template<typename T>
T thrust_reduce(const thrust::device_vector<T> *vec)
{
  return thrust::reduce(vec->begin(), vec->end(), T(0), thrust::plus<T>());
}


template<typename T>
void compare()
{
  thrust::device_vector<T> vec(1 << 28);

  thrust_reduce(&vec);
  double thrust_msecs = time_invocation_cuda(50, thrust_reduce<T>, &vec);

  my_reduce(&vec);
  double my_msecs = time_invocation_cuda(50, my_reduce<T>, &vec);

  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;

  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;
}


int main()
{
  size_t n = 123456789;

  thrust::device_vector<int> vec(n);

  thrust::sequence(vec.begin(), vec.end());

  int my_result = my_reduce(vec.begin(), vec.end(), 13, thrust::plus<int>());

  std::cout << "my_result: " << my_result << std::endl;

  int thrust_result = thrust::reduce(vec.begin(), vec.end(), 13, thrust::plus<int>());

  std::cout << "thrust_result: " << thrust_result << std::endl;

  assert(thrust_result == my_result);

  std::cout << "int: " << std::endl;
  compare<int>();

  std::cout << "long int: " << std::endl;
  compare<long int>();

  std::cout << "float: " << std::endl;
  compare<float>();

  std::cout << "double: " << std::endl;
  compare<double>();
}

