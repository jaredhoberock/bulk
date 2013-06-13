#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/sequential_executor.hpp>

BULK_NS_PREFIX
namespace bulk
{


template<std::size_t bound,
         typename RandomAccessIterator,
         typename T,
         typename BinaryFunction>
__forceinline__ __device__
T accumulate(const bounded_executor<bound> &exec,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  typedef typename bounded_executor<bound>::size_type size_type;
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  size_type n = last - first;

  #pragma unroll
  for(size_type i = 0; i < exec.bound(); ++i)
  {
    if(i < n)
    {
      init = binary_op(init, first[i]);
    } // end if
  } // end for i

  return init;
} // end accumulate()


} // end bulk

