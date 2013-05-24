#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <bulk/thread.hpp>
#include <thrust/execution_policy.h>

namespace bulk
{
namespace thread_group_detail
{



// "group" instead of "array"
// this thing is nothing like std::array or a C array
template<typename Derived, typename Thread = bulk::thread>
class thread_group_base
  : public thrust::execution_policy<Derived>
{
  public:
    typedef Thread thread_type;

    typedef unsigned int size_type;

    __host__ __device__
    thread_group_base()
      : this_thread()
    {}

    // "wait" instead of "barrier"
    // "wait" is a verb; "barrier" a noun
    // moreover "std::barrier" will likely be the name of a type in c++17
    __device__ void wait()
    {
      __syncthreads();
    }

    __device__ 
    size_type global_index() const
    {
      return derived().index() * derived().size() + this_thread.index();
    }

    __device__
    size_type index() const
    {
      return blockIdx.x;
    }

    thread_type this_thread;

  private:
    __host__ __device__
    const Derived &derived() const
    {
      return static_cast<const Derived&>(*this);
    }

    __host__ __device__
    Derived &derived()
    {
      return static_cast<Derived&>(*this);
    }
};


} // end thread_group_detail


static const std::size_t default_thread_group_size = 256;

static const std::size_t default_thread_group_grainsize = 1;


template<std::size_t size_      = default_thread_group_size,
         std::size_t grainsize_ = default_thread_group_grainsize>
class static_thread_group
  : public thread_group_detail::thread_group_base<
      static_thread_group<size_,grainsize_>
    >
{
  private:
    typedef thread_group_detail::thread_group_base<
      static_thread_group<size_,grainsize_>
    > super_t;

  public:
    typedef typename super_t::size_type size_type;

    static const size_type static_size = size_;

    static const size_type static_grainsize = grainsize_;

    __host__ __device__
    size_type size() const
    {
      return static_size;
    }

    // "grainsize" inspired by TBB
    // this is the size of the quantum of sequential work
    __device__
    size_type grainsize() const
    {
      return static_grainsize;
    }
};



class thread_group
  : public thread_group_detail::thread_group_base<thread_group>
{
  private:
    typedef thread_group_detail::thread_group_base<thread_group> super_t;

  public:
    typedef typename super_t::size_type size_type;

    __device__
    thread_group()
    {}

    explicit thread_group(size_type size)
      : m_size(size)
    {}

    __host__ __device__
    size_type size() const
    {
#ifdef __CUDA_ARCH__
      return blockDim.x;
#else
      return m_size;
#endif
    }

    __device__
    size_type grainsize() const
    {
      return 1;
    }

  private:
    size_type m_size;
};


template<typename T>
struct is_thread_group : thrust::detail::false_type {};


template<std::size_t size, std::size_t grainsize>
struct is_thread_group<static_thread_group<size,grainsize> > : thrust::detail::true_type {};


template<>
struct is_thread_group<thread_group> : thrust::detail::true_type {};


template<typename T>
struct is_static_thread_group : thrust::detail::false_type {};


template<std::size_t size, std::size_t grainsize>
struct is_static_thread_group<static_thread_group<size,grainsize> > : thrust::detail::true_type {};


template<typename T, typename U = void>
struct enable_if_thread_group
  : thrust::detail::enable_if<
      is_thread_group<T>::value,
      U
    >
{};


template<typename T, typename U = void>
struct enable_if_static_thread_group
  : thrust::detail::enable_if<
      is_static_thread_group<T>::value,
      U
    >
{};


template<typename T, typename U = void>
struct disable_if_static_thread_group
  : thrust::detail::disable_if<
      is_static_thread_group<T>::value,
      U
    >
{};


namespace detail
{


struct placeholder {};


} // end detail


static const detail::placeholder there;


} // end bulk

