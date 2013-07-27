#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_policy.hpp>


BULK_NS_PREFIX
namespace bulk
{


template<typename ExecutionGroup,
         typename RandomAccessIterator,
         typename Size,
         typename Function>
__device__
RandomAccessIterator for_each_n(ExecutionGroup &g, RandomAccessIterator first, Size n, Function f)
{
  for(Size i = g.this_thread.index();
      i < n;
      i += g.size())
  {
    f(first[i]);
  } // end for i

  g.wait();

  return first + n;
} // end for_each()


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename Size,
         typename Function>
__device__
RandomAccessIterator for_each_n(bounded<bound, bulk::agent<grainsize> > &b,
                                RandomAccessIterator first,
                                Size n,
                                Function f)
{
  typedef typename bounded<bound, bulk::agent<grainsize> >::size_type size_type;

  for(size_type i = 0; i < bound; ++i)
  {
    if(i < n)
    {
      f(first[i]);
    } // end if
  } // end for i

  return first + n;
} // end for_each_n()
                                

} // end bulk
BULK_NS_SUFFIX

