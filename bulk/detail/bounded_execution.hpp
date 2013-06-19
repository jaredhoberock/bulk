#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/sequential_executor.hpp>

BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


template<std::size_t bound_, typename Executor>
class bounded_execution
  : public Executor
{
  public:
    typedef int size_type;

    static const size_type static_bound = bound_;

    __device__
    bounded_execution(const Executor &exec)
      : Executor(exec)
    {}

    __device__
    size_type bound() const
    {
      return static_bound;
    } // end bound()
}; // end bounded_execution


} // end detail
} // end bulk
BULK_NS_SUFFIX

