#pragma once

namespace bulk
{
namespace detail
{


__device__ unsigned int __isShared(const void *ptr)
{
  unsigned int ret;

#if __CUDA_ARCH__ >= 200
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.shared p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#  if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#  else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#  endif
#else
  ret = 0;
#endif

  return ret;
} // end __isShared()


__device__ bool is_shared(const void *ptr)
{
  return __isShared(ptr);
} // end is_shared()


__device__ bool is_global(const void *ptr)
{
#if __CUDA_ARCH__ >= 200
  return __isGlobal(ptr);
#else
  return false;
#endif
} // end is_global()


} // end detail
} // end bulk

