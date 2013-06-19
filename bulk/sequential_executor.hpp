#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/detail/bounded_execution.hpp>
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

    // XXX remove these
    __device__
    void wait() const {}

    __device__
    size_type size() const
    {
      return 1;
    } // end size()
}; // end sequential_executor


template<std::size_t bound>
class bounded_executor
  : public bulk::detail::bounded_execution<bound, bulk::sequential_executor>
{
  private:
    typedef bulk::detail::bounded_execution<bound, bulk::sequential_executor> super_t;

  public:
    __device__
    bounded_executor(const bulk::sequential_executor &exec)
      : super_t(exec)
    {}
}; // end bounded_execution


template<std::size_t b>
__device__
bounded_executor<b> bound(const bulk::sequential_executor &exec)
{
  return bounded_executor<b>(exec);
} // end bound()


} // end bulk
BULK_NS_SUFFIX

