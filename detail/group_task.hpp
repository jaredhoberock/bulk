#pragma once

#include "../malloc.hpp"
#include <thrust/detail/type_traits.h>
#include "../thread_group.hpp"
#include <thrust/detail/tuple_transform.h>


namespace bulk
{
namespace detail
{


template<typename ThreadGroup, typename Closure>
class group_task
{
  public:
    __host__ __device__
    group_task(Closure c, size_t num_dynamic_smem_bytes)
      : c(c),
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

      // instantiate a view of this thread group
      ThreadGroup this_group;

      // substitute placeholders with this_group
      substituted_arguments_type new_args = substitute_placeholders(this_group, c.arguments());

      // create a new closure with the new arguments
      closure<typename Closure::function_type, substituted_arguments_type> new_c(c.function(), new_args);

      // execute the new closure
      new_c();
    }

  private:
    template<typename T>
    struct substitutor_result
      : thrust::detail::eval_if<
          thrust::detail::is_same<T, placeholder>::value,
          thrust::detail::identity_<ThreadGroup &>,
          thrust::detail::identity_<T>
        >
    {};

    typedef typename thrust::detail::tuple_meta_transform<
      typename Closure::arguments_type,
      substitutor_result
    >::type substituted_arguments_type;

    struct substitutor
    {
      ThreadGroup &g;

      __device__
      substitutor(ThreadGroup &g)
        : g(g)
      {}

      __device__
      ThreadGroup &operator()(placeholder) const
      {
        return g;
      }

      template<typename T>
      __device__
      T &operator()(T &x) const
      {
        return x;
      }
    };

    __device__
    substituted_arguments_type substitute_placeholders(ThreadGroup &g, typename Closure::arguments_type args)
    {
      return thrust::detail::tuple_host_device_transform<substitutor_result>(args, substitutor(g));
    }

    Closure c;
    size_t num_dynamic_smem_bytes;
};


} // end detail
} // end bulk

