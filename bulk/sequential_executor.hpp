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


static const std::size_t default_grain_executor_grainsize = 1;


template<std::size_t grainsize_ = default_grain_executor_grainsize>
struct grain_executor
  : thrust::system::detail::sequential::execution_policy<grain_executor<grainsize_> >
{
  public:
    typedef unsigned int size_type;

    static const size_type static_grainsize = grainsize_;

    __device__
    size_type index() const
    {
      return threadIdx.x;
    } // end index()

    __host__ __device__
    size_type grainsize() const
    {
      return static_grainsize;
    } // end grainsize()
}; // end grain_executor


}; // end bulk
BULK_NS_SUFFIX

