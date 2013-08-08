#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_policy.hpp>


BULK_NS_PREFIX
namespace bulk
{


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3>
__forceinline__ __device__
RandomAccessIterator3 gather(const bounded<bound,agent<grainsize> > &,
                             RandomAccessIterator1 map_first,
                             RandomAccessIterator1 map_last,
                             RandomAccessIterator2 input_first,
                             RandomAccessIterator3 result)
{
  typedef typename bulk::bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  size_type n = map_last - map_first;

  for(size_type i = 0; i < bound; ++i)
  {
    if(i < n)
    {
      result[i] = input_first[map_first[i]];
    }
  }

  return result + n;
} // end scatter_if()


} // end bulk
BULK_NS_SUFFIX

