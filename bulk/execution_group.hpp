#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <bulk/sequential_executor.hpp>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>

namespace bulk
{
namespace execution_group_detail
{



// "group" instead of "array"
// this thing is nothing like std::array or a C array
template<typename Derived, typename Executor = bulk::sequential_executor>
class execution_group_base
  : public thrust::execution_policy<Derived>
{
  public:
    typedef Executor executor_type;

    typedef unsigned int size_type;

    __host__ __device__
    execution_group_base()
      : this_exec()
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
      return derived().index() * derived().size() + this_exec.index();
    }

    __device__
    size_type index() const
    {
      return blockIdx.x;
    }

    executor_type this_exec;

    inline static size_type hardware_concurrency()
    {
      return static_cast<size_type>(thrust::system::cuda::detail::device_properties().multiProcessorCount);
    } // end hardware_concurrency()

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


} // end execution_group_detail


static const std::size_t default_execution_group_size = 256;

static const std::size_t default_execution_group_grainsize = 1;


template<std::size_t size_      = default_execution_group_size,
         std::size_t grainsize_ = default_execution_group_grainsize>
class static_execution_group
  : public execution_group_detail::execution_group_base<
      static_execution_group<size_,grainsize_>,
      grain_executor<grainsize_>
    >
{
  private:
    typedef execution_group_detail::execution_group_base<
      static_execution_group<size_,grainsize_>,
      grain_executor<grainsize_>
    > super_t;

    typedef typename super_t::executor_type executor_type;

  public:
    typedef typename super_t::size_type size_type;

    static const size_type static_size = size_;

    static const size_type static_grainsize = executor_type::static_grainsize;

    __host__ __device__
    size_type size() const
    {
      return static_size;
    }

    // "grainsize" inspired by TBB
    // this is the size of the quantum of sequential work
    __host__ __device__
    size_type grainsize() const
    {
      return super_t::this_exec.grainsize();
    }
};



class execution_group
  : public execution_group_detail::execution_group_base<execution_group>
{
  private:
    typedef execution_group_detail::execution_group_base<execution_group> super_t;

  public:
    typedef typename super_t::size_type size_type;

    __device__
    execution_group()
    {}

    explicit execution_group(size_type size)
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
struct is_execution_group : thrust::detail::false_type {};


template<std::size_t size, std::size_t grainsize>
struct is_execution_group<static_execution_group<size,grainsize> > : thrust::detail::true_type {};


template<>
struct is_execution_group<execution_group> : thrust::detail::true_type {};


template<typename T>
struct is_static_execution_group : thrust::detail::false_type {};


template<std::size_t size, std::size_t grainsize>
struct is_static_execution_group<static_execution_group<size,grainsize> > : thrust::detail::true_type {};


template<typename T, typename U = void>
struct enable_if_execution_group
  : thrust::detail::enable_if<
      is_execution_group<T>::value,
      U
    >
{};


template<typename T, typename U = void>
struct enable_if_static_execution_group
  : thrust::detail::enable_if<
      is_static_execution_group<T>::value,
      U
    >
{};


template<typename T, typename U = void>
struct disable_if_static_execution_group
  : thrust::detail::disable_if<
      is_static_execution_group<T>::value,
      U
    >
{};


namespace detail
{


struct placeholder {};


} // end detail


static const detail::placeholder there;


} // end bulk
