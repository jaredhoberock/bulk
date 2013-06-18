#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/detail/pointer_traits.hpp>
#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <cstdlib>


BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


extern __shared__ int s_data_segment_begin[];


class os
{
  public:
    __device__ inline os(size_t max_data_segment_size)
      : m_program_break(s_data_segment_begin),
        m_max_data_segment_size(max_data_segment_size)
    {
    }


    __device__ inline int brk(void *end_data_segment)
    {
      if(end_data_segment <= m_program_break)
      {
        m_program_break = end_data_segment;
        return 0;
      }

      return -1;
    }


    __device__ inline void *sbrk(size_t increment)
    {
      if(data_segment_size() + increment <= m_max_data_segment_size)
      {
        m_program_break = reinterpret_cast<char*>(m_program_break) + increment;
      } // end if
      else
      {
        return reinterpret_cast<void*>(-1);
      } // end else

      return m_program_break;
    }


    __device__ inline void *program_break() const
    {
      return m_program_break;
    }

    
    __device__ inline void *data_segment_begin() const
    {
      return s_data_segment_begin;
    }


  private:
    __device__ inline size_t data_segment_size()
    {
      return reinterpret_cast<char*>(m_program_break) - reinterpret_cast<char*>(s_data_segment_begin);
    } // end data_segment_size()


    void *m_program_break;

    // XXX this can safely be uint32
    size_t m_max_data_segment_size;
};


// only one instance of this class can logically exist per CTA, and its use is thread-unsafe
class singleton_unsafe_on_chip_allocator
{
  public:
    __device__ inline singleton_unsafe_on_chip_allocator(size_t max_data_segment_size)
      : m_os(max_data_segment_size)
    {}
  
    __device__ inline void *allocate(size_t size)
    {
      size_t aligned_size = align4(size);
    
      block *prev = find_first_free_insertion_point(heap_begin(), heap_end(), aligned_size);
    
      block *b;
    
      if(prev != heap_end() && (b = next(prev)) != heap_end())
      {
        // can we split?
        if((b->size - aligned_size) >= sizeof(block) + 4) // +4 for alignment
        {
          split_block(b, aligned_size);
        } // end if
    
        b->is_free = false;
      } // end if
      else
      {
        // nothing fits, extend the heap
        b = extend_heap(prev, aligned_size);
        if(b == heap_end())
        {
          return 0;
        } // end if
      } // end else
    
      return data(b);
    } // end allocate()
  
  
    __device__ inline void deallocate(void *ptr)
    {
      if(ptr != 0)
      {
        block *b = get_block(ptr);
    
        // free the block
        b->is_free = true;
    
        // try to fuse the freed block the previous block
        if(prev(b) && prev(b)->is_free)
        {
          b = prev(b);
          fuse_block(b);
        } // end if
    
        // now try to fuse with the next block
        if(next(b) != heap_end())
        {
          fuse_block(b);
        } // end if
        else
        {
          // the the OS know where the new break is
          m_os.brk(b);
        } // end else
      } // end if
    } // end deallocate()


  private:

    // XXX we could probably encode the entire block structure
    //     into a single 32b int:
    //     | 1b wasted | 1b is_free | 15b size | 15b prev (left neighbor's size) |
    //     this would make the maximum allocation 1 << 15 = 32kb
    struct block
    {
      // XXX this can safely be uint32
      size_t  size;
      block  *prev;
    
      // XXX we could use the MSB of size to encode is_free
      int     is_free;
    };
  
  
    os     m_os;

    __device__ inline block *heap_begin() const
    {
      return reinterpret_cast<block*>(m_os.data_segment_begin());
    } // end heap_begin()


    __device__ inline block *heap_end() const
    {
      return reinterpret_cast<block*>(m_os.program_break());
    } // end heap_end();

  
    __device__ inline static void *data(block *b)
    {
      return reinterpret_cast<char*>(b) + sizeof(block);
    } // end data
  
  
    __device__ inline static block *prev(block *b)
    {
      return b->prev;
    } // end prev()
  
  
    __device__ inline static block *next(block *b)
    {
      return reinterpret_cast<block*>(reinterpret_cast<char*>(data(b)) + b->size);
    } // end next()
  
  
    __device__ inline void split_block(block *b, size_t size)
    {
      block *new_block;
    
      // emplace a new block within the old one's data segment
      new_block = reinterpret_cast<block*>(reinterpret_cast<char*>(data(b)) + size);
    
      // the new block's size is the old block's size less the size of the split less the size of a block
      new_block->size = b->size - size - sizeof(block);
    
      new_block->prev = b;
      new_block->is_free = true;
    
      // the old block's size is the size of the split
      b->size = size;
    
      // link the old block to the new one
      if(next(new_block) != heap_end())
      {
        next(new_block)->prev = new_block;
      } // end if
    } // end split_block()
  
  
    __device__ inline bool fuse_block(block *b)
    {
      if(next(b) != heap_end() && next(b)->is_free)
      {
        b->size += sizeof(block) + next(b)->size;
    
        if(next(b) != heap_end())
        {
          next(b)->prev = b;
        }
    
        return true;
      }
    
      return false;
    } // end fuse_block()
  
  
    __device__ inline static block *get_block(void *data)
    {
      // the block metadata lives sizeof(block) bytes to the left of data
      return reinterpret_cast<block *>(reinterpret_cast<char *>(data) - sizeof(block));
    } // end get_block()
  
  
    __device__ inline static block *find_first_free_insertion_point(block *first, block *last, size_t size)
    {
      block *prev = last;
    
      while(first != last && !(first->is_free && first->size >= size))
      {
        prev = first;
        first = next(first);
      }
    
      return prev;
    } // end find_first_free_insertion_point()
  
  
    __device__ inline block *extend_heap(block *prev, size_t size)
    {
      // the new block goes at the current end of the heap
      block *new_block = heap_end();
    
      // move the break to the right to accomodate both a block and the requested allocation
      if(m_os.sbrk(sizeof(block) + size) == reinterpret_cast<void*>(-1))
      {
        // allocation failed
        return new_block;
      }
    
      new_block->size = size;
      new_block->prev = prev;
      new_block->is_free = false;
    
      return new_block;
    } // end extend_heap()
  
  
    __device__ inline static size_t align4(size_t size)
    {
      return ((((size - 1) >> 2) << 2) + 4);
    } // end align4()
}; // end singleton_unsafe_on_chip_allocator


class singleton_on_chip_allocator
{
  public:
    // XXX mark as __host__ to WAR a warning from uninitialized.construct
    inline __device__ __host__
    singleton_on_chip_allocator(size_t max_data_segment_size)
      : m_mutex(),
        m_alloc(max_data_segment_size)
    {}


    inline __device__
    void *unsafe_allocate(size_t size)
    {
      return m_alloc.allocate(size);
    }


    inline __device__
    void *allocate(size_t size)
    {
      void *result;

      m_mutex.lock();
      {
        result = unsafe_allocate(size);
      } // end critical section
      m_mutex.unlock();

      return result;
    } // end allocate()


    inline __device__
    void unsafe_deallocate(void *ptr)
    {
      m_alloc.deallocate(ptr);
    } // end unsafe_deallocate()


    inline __device__
    void deallocate(void *ptr)
    {
      m_mutex.lock();
      {
        unsafe_deallocate(ptr);
      } // end critical section
      m_mutex.unlock();
    } // end deallocate()


  private:
    class mutex
    {
      public:
        inline __device__
        mutex()
          : m_in_use(0)
        {}


        inline __device__
        bool try_lock()
        {
#if __CUDA_ARCH__ >= 110
          return atomicCAS(&m_in_use, 0, 1) != 0;
#else
          return false;
#endif
        } // end try_lock()


        inline __device__
        void lock()
        {
          // spin while waiting
          while(try_lock());
        } // end lock()


        inline __device__
        void unlock()
        {
          m_in_use = 0;
        } // end unlock()


      private:
        unsigned int m_in_use;
    }; // end mutex


    mutex m_mutex;
    singleton_unsafe_on_chip_allocator m_alloc;
}; // end singleton_on_chip_allocator


__shared__  thrust::system::cuda::detail::detail::uninitialized<singleton_on_chip_allocator> s_on_chip_allocator;


inline __device__ void init_on_chip_malloc(size_t max_data_segment_size)
{
  s_on_chip_allocator.construct(max_data_segment_size);
} // end init_on_chip_malloc()


template<typename T>
inline __device__ T *on_chip_cast(T *ptr)
{
  extern __shared__ char s_begin[];
  return reinterpret_cast<T*>((reinterpret_cast<char*>(ptr) - s_begin) + s_begin);
} // end on_chip_cast()


inline __device__ void *on_chip_malloc(size_t size)
{
  void *result = s_on_chip_allocator.get().allocate(size);
  return on_chip_cast(result);
} // end on_chip_malloc()


inline __device__ void on_chip_free(void *ptr)
{
  s_on_chip_allocator.get().deallocate(ptr);
} // end on_chip_free()


inline __device__ void *unsafe_on_chip_malloc(size_t size)
{
  void *result = s_on_chip_allocator.get().unsafe_allocate(size);
  return on_chip_cast(result);
} // end unsafe_on_chip_malloc()


inline __device__ void unsafe_on_chip_free(void *ptr)
{
  s_on_chip_allocator.get().unsafe_deallocate(ptr);
} // end unsafe_on_chip_free()


} // end detail


inline __device__ void *shmalloc(size_t num_bytes)
{
  // first try on_chip_malloc
  void *result = detail::on_chip_malloc(num_bytes);
  
#if __CUDA_ARCH__ >= 200
  if(!result)
  {
    result = std::malloc(num_bytes);
  } // end if
#endif // __CUDA_ARCH__

  return result;
} // end shmalloc()


inline __device__ void *unsafe_shmalloc(size_t num_bytes)
{
  // first try on_chip_malloc
  void *result = detail::unsafe_on_chip_malloc(num_bytes);
  
#if __CUDA_ARCH__ >= 200
  if(!result)
  {
    result = std::malloc(num_bytes);
  } // end if
#endif // __CUDA_ARCH__

  return result;
} // end unsafe_shmalloc()


inline __device__ void shfree(void *ptr)
{
#if __CUDA_ARCH__ >= 200
  if(bulk::detail::is_shared(ptr))
  {
    bulk::detail::on_chip_free(bulk::detail::on_chip_cast(ptr));
  } // end if
  else
  {
    std::free(ptr);
  } // end else
#else
  bulk::detail::on_chip_free(bulk::detail::on_chip_cast(ptr));
#endif
} // end shfree()


inline __device__ void unsafe_shfree(void *ptr)
{
#if __CUDA_ARCH__ >= 200
  if(bulk::detail::is_shared(ptr))
  {
    bulk::detail::unsafe_on_chip_free(bulk::detail::on_chip_cast(ptr));
  } // end if
  else
  {
    std::free(ptr);
  } // end else
#else
  bulk::detail::unsafe_on_chip_free(bulk::detail::on_chip_cast(ptr));
#endif
} // end unsafe_shfree()


template<typename ThreadGroup>
__device__
inline void *malloc(ThreadGroup &g, size_t num_bytes)
{
  __shared__ void *s_result;

  if(g.this_exec.index() == 0)
  {
    s_result = bulk::unsafe_shmalloc(num_bytes);
  } // end if

  g.wait();

  return s_result;
} // end malloc()


template<typename ThreadGroup>
__device__
inline void free(ThreadGroup &g, void *ptr)
{
  if(g.this_exec.index() == 0)
  {
    bulk::unsafe_shfree(ptr);
  } // end if

  g.wait();
} // end free()


} // end namespace bulk
BULK_NS_SUFFIX

