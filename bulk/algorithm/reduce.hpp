#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/minmax.h>
#include <bulk/malloc.hpp>

namespace bulk
{


template<typename ThreadGroup,
         typename RandomAccessIterator,
         typename T,
         typename BinaryOperation>
__device__
T reduce(ThreadGroup &exec,
         RandomAccessIterator first,
         RandomAccessIterator last,
         T init,
         BinaryOperation binary_op)
{
  T *s_sums = static_cast<int*>(bulk::malloc(exec, sizeof(T) * exec.size()));

  unsigned int i = exec.this_thread.index();
  
  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  if(i < n)
  {
    T this_sum = first[i];

    for(i += exec.size();
        i < n;
        i += exec.size())
    {
      this_sum = binary_op(this_sum, first[i]);
    } // end for i

    // XXX this should be a placement new
    s_sums[exec.this_thread.index()] = this_sum;
  } // end if

  exec.wait();

  typedef typename ThreadGroup::size_type size_type;
  size_type m = thrust::min<size_type>(n, exec.size());

  while(m > 1)
  {
    size_type half_m = m >> 1;

    if(exec.this_thread.index() < half_m)
    {
      T old_val = s_sums[exec.this_thread.index()];

      s_sums[exec.this_thread.index()] = binary_op(old_val, s_sums[m - exec.this_thread.index() - 1]);
    } // end if

    exec.wait();

    m -= half_m;
  } // end while

  exec.wait();

  T result = (n > 0) ? binary_op(init,s_sums[0]) : init;

  exec.wait();

  bulk::free(exec, s_sums);

  return result;
} // end reduce()


} // end bulk

