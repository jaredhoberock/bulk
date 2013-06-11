#pragma once

#include <bulk/detail/config.hpp>
#include <stdexcept>

BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


inline void throw_on_error(cudaError_t e, const char *message)
{
  if(e)
  {
    throw std::runtime_error(message);
  } // end if
} // end throw_on_error()


} // end detail
} // end bulk
BULK_NS_SUFFIX

