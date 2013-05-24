#pragma once

#include <bulk/thread_group.hpp>
#include <bulk/detail/is_contiguous_iterator.hpp>
#include <bulk/detail/pointer_traits.hpp>
#include <thrust/detail/type_traits.h>

namespace bulk
{
namespace detail
{


template<typename ThreadGroup,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__device__
RandomAccessIterator2 copy_n(ThreadGroup &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  for(Size i = g.this_thread.index();
      i < n;
      i += g.size())
  {
    result[i] = first[i];
  } // end for i

  g.wait();

  return result + n;
} // end copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__device__
RandomAccessIterator2 staged_copy_n(static_thread_group<size,grainsize> &g,
                                    RandomAccessIterator1 first,
                                    Size n,
                                    RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  typedef typename static_thread_group<size,grainsize>::size_type size_type;

  // stage copy through registers
  value_type stage[grainsize];

  size_type chunk_size = g.size() * grainsize;

  for(RandomAccessIterator1 last = first + n;
      first < last;
      first += chunk_size, result += chunk_size)
  {
    // avoid conditional accesses when possible
    if((last - first) >= chunk_size)
    {
      for(size_type dst_idx = 0; dst_idx < grainsize; ++dst_idx)
      {
        size_type src_idx = g.size() * dst_idx + g.this_thread.index();

        stage[dst_idx] = first[src_idx];
      } // end for dst_idx
    } // end if
    else
    {
      for(size_type dst_idx = 0; dst_idx < grainsize; ++dst_idx)
      {
        size_type src_idx = g.size() * dst_idx + g.this_thread.index();

        if(src_idx < n) stage[dst_idx] = first[src_idx];
      } // end for dst_idx
    } // end else

    // avoid conditional accesses when possible
    if((last - first) >= chunk_size)
    {
      for(size_type src_idx = 0; src_idx < grainsize; ++src_idx)
      {
        size_type dst_idx = g.size() * src_idx + g.this_thread.index();

        result[dst_idx] = stage[src_idx];
      } // end for src_idx
    } // end if
    else
    {
      for(size_type src_idx = 0; src_idx < grainsize; ++src_idx)
      {
        size_type dst_idx = g.size() * src_idx + g.this_thread.index();

        if(dst_idx < n) result[dst_idx] = stage[src_idx];
      } // end for src_idx
    } // end else
  } // end for offset

  g.wait();

  return result + n;
} // end staged_copy_n()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__device__
RandomAccessIterator2 copy_n(static_thread_group<size,grainsize> &g,
                             RandomAccessIterator1 first,
                             Size n,
                             RandomAccessIterator2 result)
{
// kepler requires staging
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 400)
  typedef thrust::detail::and_<
    bulk::detail::is_contiguous_iterator<RandomAccessIterator1>,
    bulk::detail::is_contiguous_iterator<RandomAccessIterator2>
  > both_are_contiguous;

  if(both_are_contiguous::value &&
     is_global(thrust::raw_pointer_cast(&*first)) &&
     is_shared(thrust::raw_pointer_cast(&*result)))
  {
    return staged_copy_n(g, first, n, result);
  } // end if
#endif

  return detail::copy_n<static_thread_group<size,grainsize> >(g, first, n, result);
} // end copy_n()


} // end detail


template<typename ThreadGroup,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__device__
RandomAccessIterator2 copy_n(ThreadGroup &g, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  return detail::copy_n(g, first, n, result);
} // end copy_n()


} // end bulk

