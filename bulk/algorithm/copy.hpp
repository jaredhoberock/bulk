#pragma once

#include <bulk/execution_group.hpp>
#include <bulk/detail/is_contiguous_iterator.hpp>
#include <bulk/detail/pointer_traits.hpp>
#include <thrust/detail/type_traits.h>

namespace bulk
{
namespace detail
{


template<typename ExecutionGroup,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 simple_copy_n(ExecutionGroup &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  #pragma unroll
  for(Size i = g.this_thread.index();
      i < n;
      i += g.size())
  {
    result[i] = first[i];
  } // end for i

  g.wait();

  return result + n;
} // end simple_copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 simple_copy_n(bulk::static_execution_group<size,grainsize> &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  RandomAccessIterator2 return_me = result + n;

  typedef typename bulk::static_execution_group<size,grainsize>::size_type size_type;
  size_type chunk_size = size * grainsize;

  size_type tid = g.this_thread.index();

  // XXX i have a feeling the indexing could be rewritten to require less arithmetic
  for(RandomAccessIterator1 last = first + n;
      first < last;
      first += chunk_size, result += chunk_size)
  {
    // avoid conditional accesses when possible
    if((last - first) >= chunk_size)
    {
      #pragma unroll
      for(size_type i = 0; i < grainsize; ++i)
      {
        size_type idx = size * i + tid;
        result[idx] = first[idx];
      } // end for
    } // end if
    else
    {
      #pragma unroll
      for(size_type i = 0; i < grainsize; ++i)
      {
        size_type idx = size * i + tid;
        if(idx < (last - first))
        {
          result[idx] = first[idx];
        } // end if
      } // end for
    } // end else
  } // end for

  g.wait();

  return return_me;
} // end simple_copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
void staged_copy_n(static_execution_group<size,grainsize> &g,
                   RandomAccessIterator1 first,
                   Size n,
                   RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  // XXX using int is noticeably faster than size_type (which is unsigned int)
  //typedef typename bulk::static_execution_group<size,grainsize>::size_type size_type;
  typedef int size_type;

  // stage copy through registers
  value_type stage[grainsize];

  size_type tid = g.this_thread.index();

  size_type chunk_size = size * grainsize;

  for(RandomAccessIterator1 last = first + n;
      first < last;
      first += chunk_size, result += chunk_size)
  {
    // avoid conditional accesses when possible
    if(chunk_size <= (last - first))
    {
      #pragma unroll
      for(size_type dst_idx = 0; dst_idx < grainsize; ++dst_idx)
      {
        size_type src_idx = size * dst_idx + tid;
        stage[dst_idx] = first[src_idx];
      } // end for

      #pragma unroll
      for(size_type src_idx = 0; src_idx < grainsize; ++src_idx)
      {
        result[size * src_idx + tid] = stage[src_idx];
      } // end for
    } // end if
    else
    {
      #pragma unroll
      for(size_type dst_idx = 0; dst_idx < grainsize; ++dst_idx) 
      {
        size_type src_idx = size * dst_idx + tid;
        if(src_idx < (last - first))
        {
          stage[dst_idx] = first[src_idx];
        } // end if
      } // end for

      #pragma unroll
      for(size_type src_idx = 0; src_idx < grainsize; ++src_idx)
      {
        size_type dst_idx = size * src_idx + tid;
        if(dst_idx < (last - first)) 
        {
          result[dst_idx] = stage[src_idx];
        } // end if
      } // end for
    } // end else
  } // end for

  g.wait();
} // end staged_copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 copy_n(static_execution_group<size,grainsize> &g,
                             RandomAccessIterator1 first,
                             Size n,
                             RandomAccessIterator2 result)
{
  return detail::simple_copy_n(g, first, n, result);
} // end copy_n()


} // end detail


template<typename ExecutionGroup,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__forceinline__ __device__
RandomAccessIterator2 copy_n(ExecutionGroup &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  return detail::copy_n(g, first, n, result);
} // end copy_n()


} // end bulk

