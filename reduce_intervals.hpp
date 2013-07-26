#pragma once

#include <bulk/bulk.hpp>
#include "decomposition.hpp"

struct reduce_intervals_kernel
{
  template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename BinaryFunction>
  __device__ void operator()(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &this_group,
                             RandomAccessIterator1 first,
                             Decomposition decomp,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

    typename Decomposition::range rng = decomp[this_group.index()];

    value_type init = first[rng.second-1];

    value_type sum = bulk::reduce(this_group, first + rng.first, first + rng.second - 1, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      result[this_group.index()] = sum;
    } // end if
  } // end operator()
}; // end reduce_intervals_kernel


template<typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename BinaryFunction>
RandomAccessIterator2 reduce_intervals(RandomAccessIterator1 first, Decomposition decomp, RandomAccessIterator2 result, BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type result_type;
  const size_t groupsize = 128;
  size_t heap_size = groupsize * sizeof(result_type);
  bulk::async(bulk::grid<groupsize,7>(decomp.size(),heap_size), reduce_intervals_kernel(), bulk::root.this_exec, first, decomp, result, binary_op);

  return result + decomp.size();
} // end reduce_intervals()


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename BinaryFunction>
RandomAccessIterator2 reduce_intervals(RandomAccessIterator1 first, RandomAccessIterator1 last, Size interval_size, RandomAccessIterator2 result, BinaryFunction binary_op)
{
  return reduce_intervals(first, make_blocked_decomposition<Size>(last - first,interval_size), result, binary_op);
} // end reduce_intervals()

