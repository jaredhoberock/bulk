#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/uninitialized.hpp>
#include <bulk/detail/cuda_task.hpp>
#include <bulk/detail/throw_on_error.hpp>
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/synchronize.h>

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


#if BULK_ASYNC_USE_UNINITIALIZED
// XXX uninitialized is a performance hazard
//     disable it for the moment
template<typename Function>
__global__
void launch_by_value(uninitialized<Function> f)
{
  f.get()();
}
#else
template<typename Function>
__global__
void launch_by_value(Function f)
{
  f();
}
#endif


template<typename Function>
__global__
void launch_by_pointer(const Function *f)
{
  // copy to registers
  Function f_reg = *f;
  f_reg();
}


bool verbose = false;


template<typename ExecutionGroup, typename Closure>
struct cuda_launcher_base
{
  typedef cuda_task<ExecutionGroup, Closure> task_type;

#if BULK_ASYNC_USE_UNINITIALIZED
  typedef void (*global_function_t)(uninitialized<task_type>);
#else
  typedef void (*global_function_t)(task_type);
#endif


  void launch(size_t num_blocks, size_t block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
  {
#if BULK_ASYNC_USE_UNINITIALIZED
    uninitialized<task_type> wrapped_task;
    wrapped_task.construct(task);

    get_global_function()<<<(unsigned int)num_blocks, (unsigned int)block_size, num_dynamic_smem_bytes, stream>>>(wrapped_task);
#else
    get_global_function()<<<(unsigned int)num_blocks, (unsigned int)block_size, num_dynamic_smem_bytes, stream>>>(task);
#endif

    thrust::system::cuda::detail::synchronize_if_enabled("bulk_kernel_by_value");

    if(verbose)
    {
      std::cout << "cuda_launcher_base::launch_task(): occupancy: " << maximum_potential_occupancy(get_global_function(), block_size, num_dynamic_smem_bytes) << std::endl;
      std::cout << "cuda_launcher_base::launch_task(): num_dynamic_smem_bytes: " << num_dynamic_smem_bytes << std::endl;
    } // end if
  } // end launch()


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
    if(requested_size != use_default)
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

    if(result == use_default)
    {
      function_attributes_t attr = function_attributes();

      return thrust::system::cuda::detail::block_size_with_maximum_potential_occupancy(attr, device_properties());
    } // end if

    return result;
  } // end choose_group_size()
}; // end cuda_launcher_base


template<typename ExecutionGroup, typename Closure> class cuda_launcher;


template<std::size_t gridsize, std::size_t blocksize, std::size_t grainsize, typename Closure>
struct cuda_launcher<
  parallel_group<
    concurrent_group<
      sequential_executor<grainsize>,
      blocksize
    >,
    gridsize
  >,
  Closure
>
  : public cuda_launcher_base<typename cuda_grid<gridsize,blocksize,grainsize>::type,Closure>
{
  typedef cuda_launcher_base<typename cuda_grid<gridsize,blocksize,grainsize>::type,Closure> super_t;

  typedef typename cuda_grid<gridsize,blocksize,grainsize>::type grid_type;
  typedef typename grid_type::executor_type                      block_type;
  typedef typename block_type::executor_type                     thread_type;

  typedef typename super_t::task_type task_type;

  void launch(grid_type request, Closure c, cudaStream_t stream)
  {
    grid_type g = configure(request);

    size_t num_blocks = g.size();
    size_t block_size = g.this_exec.size();
    size_t heap_size  = g.this_exec.heap_size();

    if(num_blocks > 0 && block_size > 0)
    {
      task_type task(g, c);

      super_t::launch(num_blocks, block_size, heap_size, stream, task);
    } // end if
  } // end go()

  static grid_type configure(grid_type g)
  {
    size_t block_size = super_t::choose_group_size(g.this_exec.size());
    size_t heap_size  = super_t::choose_heap_size(block_size, g.this_exec.heap_size());
    size_t num_blocks = g.size();

    return make_grid<grid_type>(num_blocks, make_block<block_type>(block_size, heap_size));
  } // end configure()
}; // end cuda_launcher


template<std::size_t groupsize, std::size_t grainsize, typename Closure>
struct cuda_launcher<
  parallel_group<
    sequential_executor<grainsize>,
    groupsize
  >,
  Closure
>
  : public cuda_launcher_base<parallel_group<sequential_executor<grainsize>,groupsize>,Closure>
{
  typedef cuda_launcher_base<parallel_group<sequential_executor<grainsize>,groupsize>,Closure> super_t;
  typedef typename super_t::task_type task_type;

  typedef parallel_group<sequential_executor<grainsize>,groupsize> group_type;

  void launch(group_type g, Closure c, cudaStream_t stream)
  {
    size_t num_blocks, block_size;
    thrust::tie(num_blocks,block_size) = configure(g);

    if(num_blocks > 0 && block_size > 0)
    {
      task_type task(g, c);

      super_t::launch(num_blocks, block_size, 0, stream, task);
    } // end if
  } // end go()

  static thrust::tuple<size_t,size_t> configure(group_type g)
  {
    size_t block_size = thrust::min<size_t>(g.size(), super_t::choose_group_size(use_default));
    size_t num_blocks = (g.size() + block_size - 1) / block_size;

    return thrust::make_tuple(num_blocks, block_size);
  } // end configure()
}; // end cuda_launcher


} // end detail
} // end bulk
BULK_NS_SUFFIX

