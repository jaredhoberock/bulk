#pragma once
#include <bulk/detail/config.hpp>
#include <bulk/future.hpp>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/minmax.h> // XXX WAR missing #include in runtime_introspection.h
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <cstddef>


BULK_NS_PREFIX
namespace bulk
{

// Executor requirements:
//
// template<typename T>
// concept bool Executor()
// {
//   return requires(T t)
//   {
//     typename T::size_type;
//     {t.index()} -> typename T::size_type;
//   }
// };
//
// ExecutionGroup requirements:
//
// template<typename T>
// concept bool ExecutionGroup()
// {
//   return Executor<T>
//       && requires(T g)
//   {
//     typename T::executor_type;
//     Executor<typename T::executor_type>();
//     {g.size()} -> typename T::size_type;
//     {g.this_exec} -> typename T::executor_type &
//   }
// };


static const int invalid_index = INT_MAX;


// sequential execution with a grainsize hint and index within a group
template<std::size_t grainsize_ = 1>
class sequential_executor
{
  public:
    typedef int size_type;

    static const size_type static_grainsize = grainsize_;

    __host__ __device__
    sequential_executor(size_type i = invalid_index)
      : m_index(i)
    {}

    __host__ __device__
    size_type index() const
    {
      return m_index;
    }

    __host__ __device__
    size_type grainsize() const
    {
      return static_grainsize;
    }

  private:
    const size_type m_index;
};


static const int use_default = INT_MAX;

static const int dynamic_group_size = 0;


namespace detail
{
namespace group_detail
{


template<typename Executor, std::size_t size_>
class group_base
{
  public:
    typedef Executor executor_type;

    typedef int size_type;

    static const size_type static_size = size_;

    __host__ __device__
    group_base(executor_type exec = executor_type(), size_type i = invalid_index)
      : this_exec(exec),
        m_index(i)
    {}

    __host__ __device__
    size_type index() const
    {
      return m_index;
    }

    __host__ __device__
    size_type size() const
    {
      return static_size;
    }

    __device__
    size_type global_index() const
    {
      return index() * size() + this_exec.index();
    }

    executor_type this_exec;

  private:
    const size_type m_index;
};


template<typename Executor>
class group_base<Executor,dynamic_group_size>
{
  public:
    typedef Executor executor_type;

    typedef int size_type;

    __host__ __device__
    group_base(size_type sz = use_default, executor_type exec = executor_type(), size_type i = invalid_index)
      : this_exec(exec),
        m_size(sz),
        m_index(i)
    {}

    __host__ __device__
    size_type index() const
    {
      return m_index;
    }

    __host__ __device__
    size_type size() const
    {
      return m_size;
    }

    __host__ __device__
    size_type global_index() const
    {
      return index() * size() + this_exec.index();
    }

    executor_type this_exec;

  private:
    const size_type m_size;
    const size_type m_index;
};


} // end group_detail
} // end detail


// a group of independent Executors
template<typename Executor = sequential_executor<>,
         std::size_t size_ = dynamic_group_size>
class parallel_group
  : public detail::group_detail::group_base<Executor,size_>
{
  private:
    typedef detail::group_detail::group_base<
      Executor,
      size_
    > super_t;

  public:
    typedef typename super_t::executor_type executor_type;

    typedef typename super_t::size_type     size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    parallel_group(executor_type exec = executor_type(), size_type i = invalid_index)
      : super_t(exec,i)
    {}
};


template<typename Executor>
class parallel_group<Executor,dynamic_group_size>
  : public detail::group_detail::group_base<Executor,dynamic_group_size>
{
  private:
    typedef detail::group_detail::group_base<
      Executor,
      dynamic_group_size
    > super_t;

  public:
    typedef typename super_t::executor_type executor_type;

    typedef typename super_t::size_type     size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    parallel_group(size_type size = use_default, executor_type exec = executor_type(), size_type i = invalid_index)
      : super_t(size,exec,i)
    {}
};


// shorthand for creating a parallel_group of sequential_executors
inline __host__ __device__
parallel_group<> par(size_t size)
{
  return parallel_group<>(size);
}


// shorthand for creating a parallel_group of Executors
template<typename Executor>
__host__ __device__
parallel_group<Executor> par(Executor exec, size_t size)
{
  return parallel_group<Executor>(size, exec);
}


template<typename Executor>
class async_launch
{
  public:
    __host__ __device__
    async_launch(Executor exec, cudaStream_t s, cudaEvent_t be = 0)
      : owns_stream(false),e(exec),s(s),be(be)
    {}

    __host__
    async_launch(Executor exec, cudaEvent_t be)
      : owns_stream(true),e(exec),s(0),be(be)
    {
      bulk::detail::future_detail::throw_on_error(cudaStreamCreate(&s), "cudaStreamCreate in async_launch ctor");
    }

    // ensure that copies never destroy the stream
    // XXX maybe we should make this type move only, or explicitly copy streams 
    __host__ __device__
    async_launch(const async_launch &other)
      : owns_stream(false),e(other.e),s(other.s),be(other.be)
    {}

    __host__ __device__
    ~async_launch()
    {
#ifndef __CUDA_ARCH__
      if(owns_stream)
      {
        // swallow the error
        cudaError_t error = cudaStreamDestroy(s);

        if(error)
        {
          std::cerr << "CUDA error after cudaStreamDestroy in async_launch dtor: " << cudaGetErrorString(error) << std::endl;
        }
      }
#endif
    }

    __host__ __device__
    Executor exec() const
    {
      return e;
    }

    __host__ __device__
    cudaStream_t stream() const
    {
      return s;
    }

    __host__ __device__
    cudaEvent_t before_event() const
    {
      return be;
    }

  private:
    bool owns_stream;
    Executor e;
    cudaEvent_t be;
    cudaStream_t s;
};


__host__ __device__
async_launch<bulk::parallel_group<> > par(cudaStream_t s, size_t num_threads)
{
  return async_launch<bulk::parallel_group<> >(bulk::parallel_group<>(num_threads), s);
}


async_launch<bulk::parallel_group<> > par(bulk::future<void> &before, size_t num_threads)
{
  cudaEvent_t before_event = bulk::detail::future_core_access::event(before);

  return async_launch<bulk::parallel_group<> >(bulk::parallel_group<>(num_threads), before_event);
}


// a group of concurrent Executors which may synchronize
template<typename Executor      = sequential_executor<>,
         std::size_t size_      = dynamic_group_size>
class concurrent_group
  : public parallel_group<Executor,size_>
{
  private:
    typedef parallel_group<
      Executor,
      size_
    > super_t;

  public:
    typedef typename super_t::executor_type executor_type;
    typedef typename super_t::size_type     size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    concurrent_group(size_type heap_size = use_default, executor_type exec = executor_type(), size_type i = invalid_index)
      : super_t(exec,i),
        m_heap_size(heap_size)
    {}

    __device__
    void wait() const
    {
      __syncthreads();
    }

    __host__ __device__
    size_type heap_size() const
    {
      return m_heap_size;
    }

    // XXX this should go elsewhere
    inline static size_type hardware_concurrency()
    {
      return static_cast<size_type>(thrust::system::cuda::detail::device_properties().multiProcessorCount);
    } // end hardware_concurrency()

  private:
    size_type m_heap_size;
};


template<typename Executor>
class concurrent_group<Executor,dynamic_group_size>
  : public parallel_group<Executor,dynamic_group_size>
{
  private:
    typedef parallel_group<
      Executor,
      dynamic_group_size
    > super_t;

  public:
    typedef typename super_t::executor_type executor_type;

    typedef typename super_t::size_type     size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    concurrent_group(size_type size = use_default,
                     size_type heap_size = use_default,
                     executor_type exec = executor_type(),
                     size_type i = invalid_index)
      : super_t(size,exec,i),
        m_heap_size(heap_size)
    {}

    __device__
    void wait()
    {
      __syncthreads();
    }

    __host__ __device__
    size_type heap_size() const
    {
      return m_heap_size;
    }

    // XXX this should go elsewhere
    inline static size_type hardware_concurrency()
    {
      return static_cast<size_type>(thrust::system::cuda::detail::device_properties().multiProcessorCount);
    } // end hardware_concurrency()

  private:
    size_type m_heap_size;
};


// shorthand for creating a concurrent_group of sequential_executors
inline __host__ __device__
concurrent_group<> con(size_t size, size_t heap_size = use_default)
{
  return concurrent_group<>(size,heap_size);
}


// shorthand for creating a concurrent_group of Executors
template<typename Executor>
__host__ __device__
concurrent_group<Executor> con(Executor exec, size_t size, size_t heap_size = use_default)
{
  return concurrent_group<Executor>(size,heap_size,exec);
}


// shorthand for creating a concurrent_group of sequential_executors with static sizing
template<std::size_t groupsize, std::size_t grainsize>
__host__ __device__
concurrent_group<bulk::sequential_executor<grainsize>,groupsize>
con(size_t heap_size)
{
  return concurrent_group<bulk::sequential_executor<grainsize>,groupsize>(heap_size);
}


// a way to statically bound the size of an Executor's work
template<std::size_t bound_, typename Executor>
class bounded_executor
  : public Executor
{
  public:
    typedef typename Executor::size_type size_type;

    static const size_type static_bound = bound_;

    __host__ __device__
    size_type bound() const
    {
      return static_bound;
    }


    __host__ __device__
    Executor &unbound()
    {
      return *this;
    }


    __host__ __device__
    const Executor &unbound() const
    {
      return *this;
    }


  private:
    // XXX delete these unless we find a need for them
    bounded_executor();

    bounded_executor(const bounded_executor &);
};


template<std::size_t bound_, typename Executor>
__host__ __device__
bounded_executor<bound_, Executor> &bound(Executor &exec)
{
  return static_cast<bounded_executor<bound_, Executor>&>(exec);
}


template<std::size_t bound_, typename Executor>
__host__ __device__
const bounded_executor<bound_, Executor> &bound(const Executor &exec)
{
  return static_cast<const bounded_executor<bound_, Executor>&>(exec);
}


namespace detail
{


template<unsigned int depth, typename Executor>
struct executor_at_depth
{
  typedef typename executor_at_depth<
    depth-1,Executor
  >::type parent_executor_type;

  typedef typename parent_executor_type::executor_type type;
};


template<typename Executor>
struct executor_at_depth<0,Executor>
{
  typedef Executor type;
};


template<typename Cursor, typename ExecutionGroup>
struct cursor_result
{
  typedef typename executor_at_depth<Cursor::depth,ExecutionGroup>::type & type;
};


template<unsigned int d> struct cursor;


template<unsigned int d>
struct cursor
{
  static const unsigned int depth = d;

  __host__ __device__ cursor() {}

  cursor<depth+1> this_exec;

  template<typename ExecutionGroup>
  static __host__ __device__
  typename cursor_result<cursor,ExecutionGroup>::type
  get(ExecutionGroup &root)
  {
    return cursor<depth-1>::get(root.this_exec);
  }
};


template<> struct cursor<3>
{
  static const unsigned int depth = 3;

  __host__ __device__ cursor() {}

  template<typename ExecutionGroup>
  static __host__ __device__
  typename cursor_result<cursor,ExecutionGroup>::type
  get(ExecutionGroup &root)
  {
    return cursor<depth-1>::get(root.this_exec);
  }
};


template<> struct cursor<0>
{
  static const unsigned int depth = 0;

  __host__ __device__ cursor() {}

  cursor<1> this_exec;

  // the root level cursor simply returns the root
  template<typename Executor>
  static __host__ __device__
  Executor &get(Executor &root)
  {
    return root;
  }
};


template<typename T> struct is_cursor : thrust::detail::false_type {};


template<unsigned int d>
struct is_cursor<cursor<d> >
  : thrust::detail::true_type
{};


} // end detail


static const detail::cursor<0> root;


// shorthand for creating a parallel group of concurrent groups of sequential_executors
inline __host__ __device__
parallel_group<concurrent_group<> > grid(size_t num_groups, size_t group_size, size_t heap_size = use_default)
{
  return par(con(group_size,heap_size), num_groups);
}


template<std::size_t groupsize, std::size_t grainsize>
__host__ __device__
parallel_group<
  concurrent_group<
    bulk::sequential_executor<grainsize>,
    groupsize
  >
>
  grid(size_t num_groups, size_t heap_size = use_default)
{
  return par(con<groupsize,grainsize>(heap_size), num_groups);
}


} // end bulk
BULK_NS_SUFFIX

