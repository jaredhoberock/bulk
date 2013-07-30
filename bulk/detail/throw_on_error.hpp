#pragma once

#include <bulk/detail/config.hpp>
#include <thrust/system/cuda/error.h>

BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


inline void throw_on_error(cudaError_t e, const char *message)
{
  if(e)
  {
    throw thrust::system_error(e, thrust::cuda_category(), message);
  } // end if
} // end throw_on_error()


} // end detail
} // end bulk
BULK_NS_SUFFIX

