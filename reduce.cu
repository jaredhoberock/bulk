#include <cstdio>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cassert>
#include <iostream>
#include "time_invocation_cuda.hpp"
#include "decomposition.hpp"


struct reduce_partitions
{
  template<typename ConcurrentGroup, typename Iterator1, typename Iterator2, typename T, typename BinaryOperation>
  __device__
  void operator()(ConcurrentGroup &this_group, Iterator1 first, Iterator1 last, Iterator2 result, T init, BinaryOperation binary_op)
  {
    T sum = bulk::reduce(this_group, first, last, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      *result = sum;
    }
  }

  template<typename ConcurrentGroup, typename Iterator1, typename Iterator2, typename BinaryOperation>
  __device__
  void operator()(ConcurrentGroup &this_group, Iterator1 first, Iterator1 last, Iterator2 result, BinaryOperation binary_op)
  {
    // noticeably faster to pass the last element as the init
    typename thrust::iterator_value<Iterator2>::type init = last[-1];
    (*this)(this_group, first, last - 1, result, init, binary_op);
  }


  template<typename ConcurrentGroup, typename Iterator1, typename Decomposition, typename Iterator2, typename T, typename BinaryFunction>
  __device__
  void operator()(ConcurrentGroup &this_group, Iterator1 first, Decomposition decomp, Iterator2 result, T init, BinaryFunction binary_op)
  {
    typename Decomposition::range range = decomp[this_group.index()];

    Iterator1 last = first + range.second;
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

  const size_type groupsize = 128;
  const size_type grainsize = 9;
  const size_type tile_size = groupsize * grainsize;
  const size_type num_tiles = (n + tile_size - 1) / tile_size;
  const size_type subscription = 10;

  bulk::concurrent_group<
    bulk::agent<grainsize>,
    groupsize
  > g;

  const size_type num_groups = thrust::min<size_type>(subscription * g.hardware_concurrency(), num_tiles);

  aligned_decomposition<size_type> decomp(n, num_groups, tile_size);

  thrust::cuda::tag t;
  thrust::detail::temporary_array<T,thrust::cuda::tag> partial_sums(t, decomp.size());

  // reduce into partial sums
  bulk::async(bulk::par(g, decomp.size()), reduce_partitions(), bulk::root.this_exec, first, decomp, partial_sums.begin(), init, binary_op);

  if(partial_sums.size() > 1)
  {
    // reduce the partial sums
    bulk::async(g, reduce_partitions(), bulk::root, partial_sums.begin(), partial_sums.end(), partial_sums.begin(), binary_op);
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

