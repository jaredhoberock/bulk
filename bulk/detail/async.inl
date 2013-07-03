#include <bulk/detail/config.hpp>
#include <bulk/async.hpp>
#include <bulk/detail/closure.hpp>
#include <bulk/detail/group_task.hpp>
#include <bulk/detail/throw_on_error.hpp>

#include <thrust/detail/config.h>
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/cuda_launch_config.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/detail/util/blocking.h>

BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


template<typename Function>
size_t maximum_potential_occupancy(Function kernel, size_t num_threads, size_t num_smem_bytes)
{
  namespace ns = thrust::system::cuda::detail;

  ns::function_attributes_t attr = ns::function_attributes(kernel);

  return ns::cuda_launch_config_detail::max_active_blocks_per_multiprocessor(ns::device_properties(),
                                                                             attr,
                                                                             num_threads,
                                                                             0);
}

using thrust::system::cuda::detail::detail::uninitialized;


// XXX uninitialized is a performance hazard
//     disable it for the moment
//template<typename Function>
//__global__
//void launch_by_value(uninitialized<Function> f)
//{
//  f.get()();
//}


template<typename Function>
__global__
void launch_by_value(Function f)
{
  f();
}


template<typename Function>
__global__
void launch_by_pointer(const Function *f)
{
  // copy to registers
  Function f_reg = *f;
  f_reg();
}


bool verbose = false;


template<typename ThreadGroup, typename Closure>
struct launcher
{
  typedef group_task<ThreadGroup, Closure> task_type;

  //typedef void (*global_function_t)(uninitialized<task_type>);
  typedef void (*global_function_t)(task_type);

  template<typename LaunchConfig>
  future<void> go(LaunchConfig l, Closure c)
  {
    size_t num_groups = l.num_groups();
    size_t group_size = choose_group_size(l.num_threads_per_group());
    size_t heap_size = choose_heap_size(group_size, l.num_smem_bytes_per_group());

    if(num_groups > 0 && group_size > 0)
    {
      task_type task(c, heap_size);

      //uninitialized<task_type> wrapped_task;
      //wrapped_task.construct(task);

      // XXX this business is pretty expensive
      //     try to avoid it or speed it up or something
      if(l.stream() != 0)
      {
        cudaEvent_t before_event;
        throw_on_error(cudaEventCreateWithFlags(&before_event, cudaEventDisableTiming | cudaEventBlockingSync), "cudaEventCreateWithFlags in launcher::go");
        throw_on_error(cudaStreamWaitEvent(l.stream(), before_event, 0), "cudaStreamWaitEvent in launcher::go");
        throw_on_error(cudaEventDestroy(before_event), "cudaEventDestroy in launcher::go");
      } // end if

      launch_by_value<<<
        static_cast<unsigned int>(num_groups),
        static_cast<unsigned int>(group_size),
        heap_size,
        l.stream()
      >>>(task);
      //>>>(wrapped_task);

      thrust::system::cuda::detail::synchronize_if_enabled("bulk_kernel_by_value");

      if(verbose)
      {
        std::cout << "async(): occupancy: " << maximum_potential_occupancy(get_global_function(), l.num_threads_per_group(), l.num_smem_bytes_per_group()) << std::endl;
      } // end if
    } // end if

    // XXX this business is pretty expensive
    //     try to avoid it or speed it up or something
    // XXX we need to think more carefully about how the events get created here
    return (l.stream() != 0) ? future_core_access::create_in_stream(l.stream()) : future<void>();
  } // end go()


  static global_function_t get_global_function()
  {
    return launch_by_value<task_type>;
  } // end get_launch_function()


  typedef thrust::system::cuda::detail::function_attributes_t function_attributes_t;

  static function_attributes_t function_attributes()
  {
    return thrust::system::cuda::detail::function_attributes(get_global_function());
  } // end function_attributes()


  typedef thrust::system::cuda::detail::device_properties_t device_properties_t;

  static device_properties_t device_properties()
  {
    return thrust::system::cuda::detail::device_properties();
  } // end device_properties()


  static size_t max_active_blocks_per_multiprocessor(const device_properties_t &props,
                                                     const function_attributes_t &attr,
                                                     size_t num_threads_per_block,
                                                     size_t num_smem_bytes_per_block)
  {
    return thrust::system::cuda::detail::cuda_launch_config_detail::max_active_blocks_per_multiprocessor(props, attr, num_threads_per_block, num_smem_bytes_per_block);
  } // end max_active_blocks_per_multiprocessor()


  // returns the maximum number of additional dynamic smem bytes that would not lower the kernel's occupancy
  static size_t dynamic_smem_occupancy_limit(device_properties_t &props, function_attributes_t &attr, size_t num_threads_per_block, size_t num_smem_bytes_per_block)
  {
    // figure out the kernel's occupancy with 0 bytes of dynamic smem
    size_t occupancy = max_active_blocks_per_multiprocessor(props, attr, num_threads_per_block, num_smem_bytes_per_block);

    return thrust::system::cuda::detail::proportional_smem_allocation(props, attr, occupancy);
  } // end smem_occupancy_limit()


  static size_t choose_heap_size(size_t group_size, size_t requested_size)
  {
    function_attributes_t attr = function_attributes();

    // if the kernel's ptx version is < 200, we return 0 because there is no heap
    // if the user requested no heap, give him no heap
    if(attr.ptxVersion < 20 || requested_size == 0)
    {
      return 0;
    } // end if

    // how much smem could we allocate without reducing occupancy?
    device_properties_t props = device_properties();
    size_t result = dynamic_smem_occupancy_limit(props, attr, group_size, 0);

    // did the caller request a particular size?
    if(requested_size != bulk::group_launch_config<ThreadGroup>::use_default)
    {
      // add in a few bytes to the request for the heap data structure
      requested_size += 48;

      if(requested_size > result)
      {
        // the request overflows occupancy, so we might as well bump it to the next level
        result = dynamic_smem_occupancy_limit(props, attr, group_size, requested_size);
      } // end else
    } // end i

    return result;
  } // end choose_smem_size()


  static size_t choose_group_size(size_t requested_size)
  {
    size_t result = requested_size;

    if(result == bulk::group_launch_config<ThreadGroup>::use_default)
    {
      function_attributes_t attr = function_attributes();

      return thrust::system::cuda::detail::block_size_with_maximum_potential_occupancy(attr, device_properties());
    } // end if

    return result;
  } // end choose_group_size()
}; // end launcher


} // end detail


namespace detail
{


template<typename LaunchConfig, typename Closure>
future<void> async(LaunchConfig l, Closure c)
{
  detail::launcher<typename LaunchConfig::execution_group_type, Closure> launcher;
  return launcher.go(l, c);
} // end async()


} // end detail


template<typename LaunchConfig, typename Function>
future<void> async(LaunchConfig l, Function f)
{
  return detail::async(l, detail::make_closure(f));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1>
future<void> async(LaunchConfig l, Function f, Arg1 arg1)
{
  return detail::async(l, detail::make_closure(f,arg1));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9));
} // end async()


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  return detail::async(l, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10));
} // end async()


} // end bulk
BULK_NS_SUFFIX

