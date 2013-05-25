#pragma once

#include <stdexcept>

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

