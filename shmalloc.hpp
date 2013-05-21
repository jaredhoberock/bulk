#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>

namespace bulk_async
{
namespace detail
{


class mutex
{
  public:
    __device__
    mutex()
      : m_in_use(0)
    {}

    __device__
    bool try_lock()
    {
      return atomicCAS(&m_in_use, 0, 1) != 0;
    }

    __device__
    void lock()
    {
      // spin while waiting
      while(try_lock());
    }

    __device__
    void unlock()
    {
      m_in_use = 0;
    }

  private:
    unsigned int m_in_use;
};


extern __shared__ thrust::detail::uint32_t s_heap[];


class on_chip_storage
{
  public:
    __device__
    inline on_chip_storage(unsigned int num_heap_bytes)
      : m_mutex(),
        m_heap_end(0),
        m_num_free_bytes(num_heap_bytes)
    {}


    inline __device__
    void *malloc(size_t num_bytes)
    {
      void *result = 0;

      if(num_bytes > 0)
      {
        // round num_bytes up to the nearest 32b word
        size_t num_words = thrust::detail::util::divide_ri(num_bytes, sizeof(thrust::detail::uint32_t));

        size_t num_aligned_bytes = num_words * sizeof(thrust::detail::uint32_t);

        m_mutex.lock();
        {
          if(m_num_free_bytes > num_aligned_bytes)
          {
            m_num_free_bytes -= num_aligned_bytes;

            result = &s_heap[m_heap_end];

            m_heap_end += num_words;
          } // end if
        } // end critical section
        m_mutex.unlock();
      } // end if
      
      return result;
    } // end malloc()


    inline __device__
    void free(void *ptr)
    {
      // just leak it
    } // end free()


  private:
    mutex m_mutex;
    unsigned int m_heap_end;
    unsigned int m_num_free_bytes;
};


__shared__ thrust::system::cuda::detail::detail::uninitialized<on_chip_storage> s_storage;


} // end detail


__device__
inline void *shmalloc(size_t num_bytes)
{
  // first try on_chip_malloc
  void *result = detail::s_storage.get().malloc(num_bytes);

  if(!result)
  {
    result = std::malloc(num_bytes);
  } // end if

  return result;
} // end shmalloc()


__device__
inline void shfree(void *ptr)
{
  if(__isGlobal(ptr))
  {
    std::free(ptr);
  } // end if
  else
  {
    detail::s_storage.get().free(ptr);
  } // end else
} // end shfree()


} // end bulk_async

