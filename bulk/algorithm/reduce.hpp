/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/algorithm/copy.hpp>
#include <bulk/malloc.hpp>
#include <bulk/uninitialized.hpp>
#include <bulk/iterator/strided_iterator.hpp>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/minmax.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename T,
         typename BinaryFunction>
__forceinline__ __device__
T reduce(const bulk::bounded<bound,bulk::agent<grainsize> > &exec,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef typename bulk::bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  size_type n = last - first;

  for(size_type i = 0; i < exec.bound(); ++i)
  {
    if(i < n)
    {
      init = binary_op(init, first[i]);
    } // end if
  } // end for i

  return init;
} // end reduce()


namespace detail
{
namespace reduce_detail
{


template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T destructive_reduce_n(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
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

  T result = init;
  if(n > 0)
  {
    result = binary_op(result,first[0]);
  } // end if

  g.wait();

  return result;
} // end destructive_reduce_n()


} // end reduce_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T reduce(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef int size_type;

  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  T this_sum;

  bool this_sum_defined = false;

  typename thrust::iterator_traits<RandomAccessIterator>::difference_type n = last - first;

  for(; first < last; first += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);

    typedef typename thrust::iterator_value<RandomAccessIterator>::type input_type;
    
    // load input into register
    input_type local_inputs[grainsize];

    // each agent strides through the input range
    // and copies into a local array
    strided_iterator<RandomAccessIterator,size_type> local_first = make_strided_iterator(first + tid, static_cast<size_type>(groupsize));

    // XXX if we could precompute local_size for the else branch,
    //     we could just call copy_n here
    //     we can't precompute it (without a divide afaik), so we compute local_size in the else branch
    size_type local_size = 0;
    if(partition_size < elements_per_group)
    {
      size_type index = tid;
      for(size_type i = 0; i < grainsize; ++i, ++local_first, index += groupsize)
      {
        if(index < partition_size)
        {
          local_inputs[i] = *local_first;
          ++local_size;
        } // end if
      } // end for
    } // end if
    else
    {
      local_size = grainsize;
      bulk::copy_n(bulk::bound<grainsize>(g.this_exec),
                   local_first,
                   local_size,
                   local_inputs);
    } // end else

    // reduce local_inputs sequentially
    this_sum = this_sum_defined ?
      bulk::reduce(bulk::bound<grainsize>(g.this_exec), local_inputs, local_inputs + local_size, this_sum, binary_op) :
      bulk::reduce(bulk::bound<grainsize-1>(g.this_exec), local_inputs + 1, local_inputs + local_size, local_inputs[0], binary_op);

    this_sum_defined = true;
  } // end for

#if __CUDA_ARCH__ >= 200
  T *buffer = reinterpret_cast<T*>(bulk::malloc(g, groupsize * sizeof(T)));
#else
  __shared__ bulk::uninitialized_array<T,groupsize> buffer_impl;
  T *buffer = buffer_impl.data();
#endif

  if(this_sum_defined)
  {
    buffer[tid] = this_sum;
  } // end if

  g.wait();

  // reduce across the group
  T result = bulk::detail::reduce_detail::destructive_reduce_n(g, buffer, thrust::min<size_type>(groupsize,n), init, binary_op);

#if __CUDA_ARCH__ >= 200
  bulk::free(g,buffer);
#endif

  return result;
} // end reduce


template<typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T reduce(bulk::concurrent_group<> &g,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryFunction binary_op)
{
  typedef int size_type;

  const size_type groupsize = g.size();
  typedef typename bulk::concurrent_group<>::agent_type agent_type;
  const size_type grainsize = agent_type::static_grainsize;
  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  T this_sum;

  bool this_sum_defined = false;

  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  T *buffer = reinterpret_cast<T*>(bulk::malloc(g, groupsize * sizeof(T)));
  
  for(size_type input_idx = 0; input_idx < n; input_idx += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, n - input_idx);

    typedef typename thrust::iterator_value<RandomAccessIterator>::type input_type;
    
    // load input into register
    input_type inputs[grainsize];

    // XXX this is a sequential strided copy
    //     the stride is groupsize
    if(partition_size >= elements_per_group)
    {
      for(size_type i = 0; i < grainsize; ++i)
      {
        inputs[i] = first[input_idx + groupsize * i + tid];
      } // end for
    } // end if
    else
    {
      for(size_type i = 0; i < grainsize; ++i)
      {
        size_type index = groupsize * i + tid;
        if(index < partition_size)
        {
          inputs[i] = first[input_idx + index];
        } // end if
      } // end for
    } // end else
    
    // sum sequentially
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type index = groupsize * i + g.this_exec.index();
      if(index < partition_size)
      {
        input_type x = inputs[i];
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

  return result;
} // end reduce


} // end bulk
BULK_NAMESPACE_SUFFIX

