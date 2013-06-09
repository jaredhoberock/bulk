#pragma once

#include <thrust/system/detail/sequential/execution_policy.h>

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


}; // end bulk

