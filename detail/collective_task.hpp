#pragma once

#include "../shmalloc.hpp"

namespace bulk_async
{
namespace detail
{


template<typename Closure>
class collective_task
{
  public:
    __host__ __device__
    collective_task(Closure c, size_t num_dynamic_smem_bytes)
      : closure(c),
        num_dynamic_smem_bytes(num_dynamic_smem_bytes)
    {}

    __device__
    void operator()()
    {
#if __CUDA_ARCH__ >= 200
      // initialize shared storage
      if(threadIdx.x == 0)
      {
        detail::s_storage.construct(num_dynamic_smem_bytes);
      }
      __syncthreads();
#endif
      
      // execute the closure
      closure();
    }

  private:
    Closure closure;
    size_t num_dynamic_smem_bytes;
};


} // end detail
} // end bulk_async

