#include "../bulk_async.hpp"
#include "closure.hpp"
#include "collective_task.hpp"
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
  typedef collective_task<Function> task_type;

  typedef void (*global_function_t)(uninitialized<task_type>);

  void go(launch l, Function f)
  {
    l.configure(f);

    std::cout << "num_smem_bytes_per_block: " << l.num_smem_bytes_per_block() << std::endl;

    #if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
      if(l.num_blocks() > 0 && l.num_threads_per_block() > 0)
      {
        task_type task(f, l.num_smem_bytes_per_block());

        uninitialized<task_type> wrapped_task;
        wrapped_task.construct(task);

        launch_by_value<<<
          static_cast<unsigned int>(l.num_blocks()),
          static_cast<unsigned int>(l.num_threads_per_block()),
          static_cast<size_t>(l.num_smem_bytes_per_block())
        >>>(wrapped_task);

        thrust::system::cuda::detail::synchronize_if_enabled("bulk_async_kernel_by_value");
      } // end if
    #endif // __CUDA_ARCH__
    #endif // THRUST_DEVICE_COMPILER_NVCC
  } // end launch()

  static global_function_t get_global_function()
  {
    return launch_by_value<task_type>;
  } // end get_launch_function()
}; // end launcher


template<typename Function>
size_t choose_block_size(Function f)
{
  namespace ns = thrust::system::cuda::detail;

  ns::function_attributes_t attr = ns::function_attributes(launcher<Function>::get_global_function());

  return ns::block_size_with_maximum_potential_occupancy(attr, ns::device_properties());
} // end choose_block_size()


template<typename Function>
size_t choose_smem_size(size_t num_threads_per_block, Function f)
{
  namespace ns = thrust::system::cuda::detail;

  ns::function_attributes_t attr = ns::function_attributes(launcher<Function>::get_global_function());

  size_t occupancy =
    ns::cuda_launch_config_detail::max_active_blocks_per_multiprocessor(ns::device_properties(),
                                                                        attr,
                                                                        num_threads_per_block,
                                                                        0);

  return ns::proportional_smem_allocation(ns::device_properties(), attr, occupancy);
} // end choose_smem_size()


} // end detail


template<typename Function>
void launch::configure(Function f)
{
  if(m_num_threads_per_block == use_default)
  {
    m_num_threads_per_block = detail::choose_block_size(f);
    m_num_blocks = thrust::detail::util::divide_ri(m_num_threads, m_num_threads_per_block);
  } // end if

  if(m_num_smem_bytes_per_block == use_default)
  {
    m_num_smem_bytes_per_block = detail::choose_smem_size(m_num_threads_per_block, f);
  } // end if
} // end launch::configure()


template<typename Function>
void bulk_async(launch l, Function f)
{
  detail::launcher<Function> launcher;
  launcher.go(l, f);
} // end bulk_async()


template<typename Function, typename Arg1>
void bulk_async(launch l, Function f, Arg1 arg1)
{
  bulk_async(l, detail::make_closure(f,arg1));
} // end bulk_async()


template<typename Function, typename Arg1, typename Arg2>
void bulk_async(launch l, Function f, Arg1 arg1, Arg2 arg2)
{
  bulk_async(l, detail::make_closure(f,arg1,arg2));
} // end bulk_async()


template<typename Function, typename Arg1, typename Arg2, typename Arg3>
void bulk_async(launch l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  bulk_async(l, detail::make_closure(f,arg1,arg2,arg3));
} // end bulk_async()


} // end bulk_async

