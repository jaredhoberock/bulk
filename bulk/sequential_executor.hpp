#pragma once

#include <bulk/detail/config.hpp>
#include <thrust/system/detail/sequential/execution_policy.h>

BULK_NS_PREFIX
namespace bulk
{


struct sequential_executor
  : thrust::system::detail::sequential::execution_policy<sequential_executor>
{
  public:
    typedef unsigned int size_type;

    __device__
    size_type index() const
    {
      return threadIdx.x;
    } // end index()
}; // end sequential_executor


// XXX consider making this more like an adaptor
template<std::size_t bound_>
struct bounded_executor
  : thrust::system::detail::sequential::execution_policy<bounded_executor<bound_> >
{
  typedef int size_type;

  static const size_type static_bound = bound_;

  __device__
  size_type index() const
  {
    return threadIdx.x;
  } // end index()


  __device__
  size_type bound() const
  {
    return static_bound;
  } // end bound()
}; // end bounded_executor


template<std::size_t b, typename SequentialExecutor>
__device__
bounded_executor<b> bound(const SequentialExecutor &)
{
  return bounded_executor<b>();
} // end bound()


}; // end bulk
BULK_NS_SUFFIX

