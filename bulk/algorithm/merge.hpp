#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/sequential_executor.hpp>
#include <bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/detail/uninitialized.h>


BULK_NS_PREFIX
namespace bulk
{


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename Compare>
__device__
Size merge_path(RandomAccessIterator1 first1, Size n1,
                RandomAccessIterator2 first2, Size n2,
                Size diag,
                Compare comp)
{
  Size begin = thrust::max<Size>(Size(0), diag - n2);
  Size end = thrust::min<Size>(diag, n1);
  
  while(begin < end)
  {
    Size mid = (begin + end) >> 1;

    if(comp(first2[diag - 1 - mid], first1[mid]))
    {
      end = mid;
    } // end if
    else
    {
      begin = mid + 1;
    } // end else
  } // end while

  return begin;
} // end merge_path()


template<std::size_t bound,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Compare>
__device__
OutputIterator merge(const bulk::bounded_executor<bound> &e,
                     InputIterator1 first1, InputIterator1 last1,
                     InputIterator2 first2, InputIterator2 last2,
                     OutputIterator result,
                     Compare comp)
{
  typedef int size_type;

  typedef typename thrust::iterator_value<InputIterator1>::type value_type1;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type2;

  size_type n1 = last1 - first1;
  size_type idx1 = 0;

  size_type n2 = last2 - first2;
  size_type idx2 = 0;

  using thrust::system::cuda::detail::detail::uninitialized;

  uninitialized<value_type1> a;
  uninitialized<value_type2> b;

  if(n1)
  {
    a.construct(first1[0]);
  } // end if

  if(n2)
  {
    b.construct(first2[0]);
  } // end if

  size_type i = 0;
  #pragma unroll
  for(; i < bound; ++i)
  {
    // 4 cases:
    // 0. both ranges are exhausted
    // 1. range 1 is exhausted
    // 2. range 2 is exhausted
    // 3. neither range is exhausted

    const bool exhausted1 = idx1 >= n1;
    const bool exhausted2 = idx2 >= n2;

    if(exhausted1 && exhausted2)
    {
      break;
    } // end if
    else if(exhausted1)
    {
      result[i] = b;
      ++idx2;
    } // end else if
    else if(exhausted2)
    {
      result[i] = a;
      ++idx1;
    } // end else if
    else
    {
      if(!comp(b.get(),a.get()))
      {
        result[i] = a;
        ++idx1;

        if(idx1 < n1)
        {
          a = first1[idx1];
        } // end if
      } // end if
      else
      {
        result[i] = b;
        ++idx2;

        if(idx2 < n2)
        {
          b = first2[idx2];
        } // end if
      } // end else
    } // end else
  } // end for i

  if(n1)
  {
    a.destroy();
  } // end if

  if(n2)
  {
    b.destroy();
  } // end if

  return result + i;
} // end merge


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename Compare>
__device__
typename thrust::detail::enable_if<
  (bound <= groupsize * grainsize)
>::type
inplace_merge(const bulk::bounded_static_execution_group<bound,groupsize,grainsize> &g_,
              RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last,
              Compare comp)
{
  bulk::bounded_static_execution_group<bound,groupsize,grainsize> &g =
    const_cast<bulk::bounded_static_execution_group<bound,groupsize,grainsize>&>(g_);

  typedef int size_type;

  size_type n1 = middle - first;
  size_type n2 = last - middle;

  // Run a merge path to find the start of the serial merge for each thread.
  size_type local_offset = grainsize * g.this_exec.index();

  size_type mp = bulk::merge_path(first, n1, middle, n2, local_offset, comp);
  
  // Compute the ranges of the sources in shared memory.
  size_type local_offset1 = mp;
  size_type local_offset2 = n1 + local_offset - mp;
  
  // Serial merge into register.
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  value_type local_result[grainsize];
  bulk::merge(bulk::bound<grainsize>(g.this_exec),
              first + local_offset1, middle,
              first + local_offset2, last,
              local_result,
              comp);

  g.wait();

  // copy local result back to source
  // this is faster than getting the size from merge's result
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n1 + n2 - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_result, local_size, first + local_offset); 

  g.wait();
} // end inplace_merge()


} // end bulk
BULK_NS_SUFFIX

