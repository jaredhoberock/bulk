#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/sequential_executor.hpp>
#include <thrust/system/cuda/detail/detail/uninitialized.h>


BULK_NS_PREFIX
namespace bulk
{


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


} // end bulk
BULK_NS_SUFFIX

