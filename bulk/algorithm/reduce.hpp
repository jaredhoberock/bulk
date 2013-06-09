#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/minmax.h>
#include <bulk/algorithm/copy.hpp>
#include <bulk/malloc.hpp>

namespace bulk
{
namespace detail
{
namespace reduce_detail
{


template<typename ExecutionGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T destructive_reduce_n(ExecutionGroup &g, RandomAccessIterator s_sums, Size n, T init, BinaryFunction binary_op)
{
  typedef int size_type;

  size_type tid = g.this_exec.index();

  Size m = n;

  while(m > 1)
  {
    Size half_m = m >> 1;

    if(tid < half_m)
    {
      T old_val = s_sums[tid];

      s_sums[tid] = binary_op(old_val, s_sums[m - tid - 1]);
    } // end if

    g.wait();

    m -= half_m;
  } // end while

  g.wait();

  T result = (n > 0) ? binary_op(init,s_sums[0]) : init;

  g.wait();

  return result;
}


} // end reduce_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T reduce(bulk::static_execution_group<groupsize,grainsize> &g,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  typedef int size_type;

  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  value_type this_sum;

  bool this_sum_defined = false;

  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  T *buffer = reinterpret_cast<value_type*>(bulk::malloc(g, groupsize * sizeof(T)));
  
  for(; first < last; first += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // load input into register
    value_type inputs[grainsize];

    if(partition_size >= elements_per_group)
    {
      for(size_type i = 0; i < grainsize; ++i)
      {
        inputs[i] = first[groupsize * i + tid];
      } // end for
    } // end if
    else
    {
      for(size_type i = 0; i < grainsize; ++i)
      {
        size_type index = groupsize * i + tid;
        if(index < partition_size)
        {
          inputs[i] = first[index];
        } // end if
      } // end for
    } // end else
    
    // sum sequentially
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type index = groupsize * i + g.this_exec.index();
      if(index < partition_size)
      {
        value_type x = inputs[i];
        this_sum = (i || this_sum_defined) ? binary_op(this_sum, x) : x;
      } // end if
    } // end for

    this_sum_defined = true;
  } // end for

  if(this_sum_defined)
  {
    buffer[tid] = this_sum;
  } // end if

  g.wait();

  // reduce across the block
  T result = detail::reduce_detail::destructive_reduce_n(g, buffer, thrust::min<size_type>(groupsize,n), init, binary_op);

  bulk::free(g,buffer);

  g.wait();

  return result;
} // end reduce


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T noncommutative_reduce(bulk::static_execution_group<groupsize,grainsize> &g,
                        RandomAccessIterator first,
                        RandomAccessIterator last,
                        T init,
                        BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  typedef int size_type;

  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  T sum = init;

  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  union buffer_type
  {
    value_type inputs[elements_per_group];
    T sums[groupsize];
  } *buffer;

  buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));
  
  for(; first < last; first += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // copy partition into smem
    bulk::copy_n(g, first, partition_size, buffer->inputs);
    
    T this_sum;
    size_type local_offset = grainsize * g.this_exec.index();

    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type index = local_offset + i;
      if(index < partition_size)
      {
        T x = buffer->inputs[index];
        this_sum = i ? binary_op(this_sum, x) : x;
      } // end if
    } // end for

    g.wait();

    if(local_offset < partition_size)
    {
      buffer->sums[tid] = this_sum;
    } // end if

    g.wait();
    
    // sum over the group
    sum = detail::reduce_detail::destructive_reduce_n(g, buffer->sums, thrust::min<size_type>(groupsize,n), sum, binary_op);
  } // end for

  bulk::free(g, buffer);

  return sum;
} // end noncommutative_reduce


} // end bulk

