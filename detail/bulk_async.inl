#include "../bulk_async.hpp"
#include "closure.hpp"
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/cuda_launch_config.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/detail/util/blocking.h>

namespace bulk_async
{
namespace detail
{

using thrust::system::cuda::detail::detail::uninitialized;


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
template<typename Function>
__global__
void launch_by_value(uninitialized<Function> f)
{
  f.get()();
}


template<typename Function>
__global__
void launch_by_pointer(const Function *f)
{
  // copy to registers
  Function f_reg = *f;
  f_reg();
}
#else
template<typename Function>
void launch_by_value(uninitialized<Function> f) {}

template<typename Function>
void launch_by_pointer(const Function *f) {}
#endif


template<typename Function>
struct launcher
{
  typedef void (*launch_function_t)(uninitialized<Function>);

  void launch(grid_size_t num_blocks, block_size_t num_threads_per_block, Function f)
  {
    #if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
      if(num_blocks > 0 && num_threads_per_block > 0)
      {
        uninitialized<Function> wrapped_f;
        wrapped_f.construct(f);
        launch_by_value<<<num_blocks, num_threads_per_block>>>(wrapped_f);
        thrust::system::cuda::detail::synchronize_if_enabled("bulk_async_kernel_by_value");
      } // end if
    #endif // __CUDA_ARCH__
    #endif // THRUST_DEVICE_COMPILER_NVCC
  } // end launch()

  static launch_function_t get_launch_function()
  {
    return launch_by_value<Function>;
  } // end get_launch_function()
}; // end launcher


template<typename Function>
block_size_t choose_block_size(Function f)
{
  namespace ns = thrust::system::cuda::detail;

  ns::function_attributes_t attr = ns::function_attributes(launcher<Function>::get_launch_function());

  return ns::block_size_with_maximum_potential_occupancy(attr, ns::device_properties());
} // end choose_block_size()


} // end detail


template<typename DerivedPolicy,
         typename Function>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &,
                grid_size_t num_blocks,
                block_size_t num_threads_per_block,
                Function f)
{
  detail::launcher<Function> launcher;
  launcher.launch(num_blocks, num_threads_per_block, f);
} // end bulk_async()


template<typename DerivedPolicy,
         typename Function>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                size_t num_threads,
                Function f)
{
  block_size_t num_threads_per_block = detail::choose_block_size(f);
  grid_size_t num_blocks = static_cast<grid_size_t>(thrust::detail::util::divide_ri(num_threads, num_threads_per_block));
  bulk_async(exec, num_blocks, num_threads_per_block, f);
} // end bulk_async()


template<typename DerivedPolicy,
         typename Function,
         typename Arg1>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                grid_size_t num_blocks,
                block_size_t num_threads_per_block,
                Function f,
                Arg1 arg1)
{
  bulk_async(exec, num_blocks, num_threads_per_block, detail::make_closure(f,arg1));
} // end bulk_async()


template<typename DerivedPolicy,
         typename Function,
         typename Arg1>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                size_t num_threads,
                Function f,
                Arg1 arg1)
{
  bulk_async(exec, num_threads, detail::make_closure(f,arg1));
} // end bulk_async()


} // end bulk_async

