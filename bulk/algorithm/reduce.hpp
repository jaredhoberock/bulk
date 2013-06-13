#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/algorithm/copy.hpp>
#include <bulk/algorithm/accumulate.hpp>
#include <bulk/malloc.hpp>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/minmax.h>


BULK_NS_PREFIX
namespace bulk
{
namespace detail
{
namespace reduce_detail
{


template<typename ExecutionGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T destructive_reduce_n(ExecutionGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  typedef int size_type;

  size_type tid = g.this_exec.index();

  Size m = n;

  while(m > 1)
  {
    Size half_m = m >> 1;

    if(tid < half_m)
    {
      T old_val = first[tid];

      first[tid] = binary_op(old_val, first[m - tid - 1]);
    } // end if

    g.wait();

    m -= half_m;
  } // end while

  g.wait();

  T result = (n > 0) ? binary_op(init,first[0]) : init;

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

#if __CUDA_ARCH__ >= 200
  T *buffer = reinterpret_cast<value_type*>(bulk::malloc(g, groupsize * sizeof(T)));
#else
  __shared__ thrust::system::cuda::detail::detail::uninitialized_array<T,groupsize> buffer_impl;
  T *buffer = buffer_impl.data();
#endif
  
  for(; first < last; first += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // load input into register
    value_type inputs[grainsize];

    // XXX this is a sequential strided copy
    //     the stride is groupsize
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

#if __CUDA_ARCH__ >= 200
  bulk::free(g,buffer);
#endif

  return result;
} // end reduce


namespace detail
{
namespace reduce_detail
{


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T>
struct noncommutative_reduce_buffer
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  union
  {
    value_type inputs[groupsize * grainsize];
    T sums[groupsize];
  }; // end union
}; // end noncommutative_reduce_buffer


} // end reduce_detail
} // end detail


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

  typedef detail::reduce_detail::noncommutative_reduce_buffer<
    groupsize,
    grainsize,
    RandomAccessIterator,
    T
  > buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));
#else
  __shared__ thrust::system::cuda::detail::detail::uninitialized<buffer_type> buffer_impl;
  buffer_type *buffer = &buffer_impl.get();
#endif
  
  for(; first < last; first += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // copy partition into smem
    bulk::copy_n(g, first, partition_size, buffer->inputs);
    
    T this_sum;
    size_type local_offset = grainsize * g.this_exec.index();

    size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, partition_size - grainsize * tid));

    if(local_size)
    {
      this_sum = buffer->inputs[local_offset];
      this_sum = bulk::accumulate(bound<grainsize-1>(g.this_exec),
                                  buffer->inputs + local_offset + 1,
                                  buffer->inputs + local_offset + local_size,
                                  this_sum,
                                  binary_op);
    } // end if

    g.wait();

    if(local_size)
    {
      buffer->sums[tid] = this_sum;
    } // end if

    g.wait();
    
    // sum over the group
    sum = detail::reduce_detail::destructive_reduce_n(g, buffer->sums, thrust::min<size_type>(groupsize,n), sum, binary_op);
  } // end for

#if __CUDA_ARCH__ >= 200
  bulk::free(g, buffer);
#endif

  return sum;
} // end noncommutative_reduce


} // end bulk
BULK_NS_SUFFIX

