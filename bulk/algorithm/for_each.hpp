#pragma once

#include <bulk/detail/config.hpp>


BULK_NS_PREFIX
namespace bulk
{


template<typename ThreadGroup,
         typename RandomAccessIterator,
         typename Size,
         typename Function>
__device__
RandomAccessIterator for_each_n(ThreadGroup &g, RandomAccessIterator first, Size n, Function f)
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


} // end bulk
BULK_NS_SUFFIX

