#include <moderngpu.cuh>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cassert>
#include <iostream>
#include "time_invocation_cuda.hpp"
#include <thrust/detail/temporary_array.h>
#include <thrust/copy.h>
#include <bulk/bulk.hpp>


typedef int T;


template<unsigned int size, unsigned int grainsize>
struct inclusive_scan_n
{
  template<typename InputIterator, typename Size, typename OutputIterator, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<size,grainsize> &this_group, InputIterator first, Size n, OutputIterator result, BinaryFunction binary_op)
  {
    bulk::inclusive_scan(this_group, first, first + n, result, binary_op);
  }
};


template<unsigned int size, unsigned int grainsize>
struct exclusive_scan_n
{
  template<typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<size,grainsize> &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk::exclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};


template<unsigned int size, typename ThreadGroup, typename T, typename BinaryFunction>
__device__ T small_inplace_exclusive_scan_with_buffer(ThreadGroup &g, T *first, T init, T *buffer, BinaryFunction binary_op)
{
  // XXX int is noticeably faster than ThreadGroup::size_type
  typedef int size_type;
  //typedef typename ThreadGroup::size_type size_type;

  // ping points to the most current data
  T *ping = first;
  T *pong = buffer;

  size_type tid = g.this_thread.index();

  if(tid == 0)
  {
    first[0] = binary_op(init, first[0]);
  }

  T x = first[tid];

  g.wait();

  #pragma unroll
  for(size_type offset = 1; offset < size; offset += offset)
  {
    if(tid >= offset)
    {
      x = binary_op(ping[tid - offset], x);
    }

    thrust::swap(ping, pong);

    ping[tid] = x;

    g.wait();
  }

  T result = ping[size - 1];

  x = (tid == 0) ? init : ping[tid - 1];

  g.wait();

  first[tid] = x;

  g.wait();

  return result;
}


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__device__ void inclusive_scan_with_carry_with_buffer(bulk::static_thread_group<groupsize,grainsize> &g,
                                                      RandomAccessIterator1 first, RandomAccessIterator1 last,
                                                      RandomAccessIterator2 result,
                                                      T carry_in,
                                                      BinaryFunction binary_op,
                                                      void *buffer)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type  input_type;
  // XXX this needs to be inferred from the iterators and binary op
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type intermediate_type;

  intermediate_type *s_sums = reinterpret_cast<intermediate_type*>(buffer);

  union {
    input_type        *inputs;
    intermediate_type *results;
  } shared;

  shared.inputs = reinterpret_cast<intermediate_type*>(reinterpret_cast<char*>(buffer) + 2*groupsize*sizeof(intermediate_type));

  // XXX int is noticeably faster than ThreadGroup::size_type
  //typedef typename bulk::static_thread_group<groupsize,grainsize>::size_type size_type;
  typedef int size_type;

  size_type tid = g.this_thread.index();

  size_type elements_per_group = groupsize * grainsize;

  for(; first < last; first += elements_per_group, result += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // stage data through shared memory
    bulk::copy_n(g, first, partition_size, shared.inputs);
    
    // Transpose out of shared memory.
    input_type local_inputs[grainsize];

    size_type local_offset = grainsize * tid;

    size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, partition_size - grainsize * tid));

    // XXX this should be uninitialized<input_type>
    input_type x;

    // this loop is a fused copy and accumulate
    #pragma unroll
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type index = local_offset + i;
      if(index < partition_size)
      {
        local_inputs[i] = shared.inputs[index];
        x = i ? binary_op(x, local_inputs[i]) : local_inputs[i];
      } // end if
    } // end for

    if(local_size)
    {
      s_sums[tid] = x;
    } // end if

    g.wait();
    
    // exclusive scan the array of per-thread sums
    carry_in = small_inplace_exclusive_scan_with_buffer<groupsize>(g, s_sums, carry_in, s_sums + groupsize, binary_op);

    if(local_size)
    {
      x = s_sums[tid];
    } // end if
    
    // this loop is an inclusive_scan_with_carry (x begins as the carry)
    // XXX this loop should be one of the things to modify when porting to exclusive_scan
    #pragma unroll
    for(size_type i = 0; i < grainsize; ++i) 
    {
      size_type index = local_offset + i;
      if(index < partition_size)
      {
        x = binary_op(x, local_inputs[i]);

        shared.results[index] = x;
      } // end if
    } // end for

    g.wait();
    
    bulk::copy_n(g, shared.results, partition_size, result);
  } // end for
}


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__device__ void inclusive_scan_with_carry(bulk::static_thread_group<groupsize,grainsize> &g,
                                          RandomAccessIterator1 first, RandomAccessIterator1 last,
                                          RandomAccessIterator2 result,
                                          T carry_in,
                                          BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type  input_type;
  // XXX this needs to be inferred from the iterators and binary op
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type intermediate_type;

  int num_stage_bytes = groupsize * grainsize * thrust::max<int>(sizeof(input_type),sizeof(intermediate_type));
  int num_sums_bytes = 2 * groupsize * sizeof(intermediate_type);

  void *buffer = bulk::malloc(g, num_stage_bytes + num_sums_bytes);

  if(bulk::detail::is_shared(buffer))
  {
    inclusive_scan_with_carry_with_buffer(g, first, last, result, carry_in, binary_op, bulk::detail::on_chip_cast(buffer));
  }
  else
  {
    inclusive_scan_with_carry_with_buffer(g, first, last, result, carry_in, binary_op, buffer);
  }

  bulk::free(g, buffer);
} // end inclusive_scan_with_carry()


template<std::size_t groupsize, std::size_t grainsize>
struct inclusive_downsweep
{
  template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<groupsize,grainsize> &this_group,
                             RandomAccessIterator1 first,
                             int count,
                             int2 task,
                             const T *carries,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    const int elements_per_group = groupsize * grainsize;
  
    int2 range = mgpu::ComputeTaskRange(this_group.index(), task, elements_per_group, count);
    
    // give group 0 a carry by taking the first input element
    // and adjusting its range
    T carry = (this_group.index() != 0) ? carries[this_group.index()] : first[0];
    if(this_group.index() == 0)
    {
      if(this_group.this_thread.index() == 0)
      {
        *result = carry;
      }
  
      ++range.x;
    }
  
    RandomAccessIterator1 last = first + range.y;
    first += range.x;
    result += range.x;
  
    inclusive_scan_with_carry(this_group, first, last, result, carry, binary_op);
  }
};


template<mgpu::MgpuScanType Type, typename InputIt, typename OutputIt, typename Op>
void IncScan(InputIt data_global, int count, OutputIt dest_global, Op op, mgpu::CudaContext& context)
{
  typedef typename Op::value_type value_type;
  typedef typename Op::result_type result_type;
  
  const int threshold_of_parallelism = 20000;

  if(count < threshold_of_parallelism)
  {
    const int size = 512;
    const int grainsize = 3;

    bulk::static_thread_group<size,grainsize> group;
    bulk::async(bulk::par(group, 1), inclusive_scan_n<size,grainsize>(), bulk::there, data_global, count, dest_global, thrust::plus<int>());
  }
  else
  {
    // Run the parallel raking reduce as an upsweep.
    const int groupsize1 = 128;
    const int grainsize1 = 7;
    typedef mgpu::LaunchBoxVT<groupsize1, grainsize1> Tuning;
    int2 launch = Tuning::GetLaunchParams(context);
    const int NV = launch.x * launch.y;
    
    int numTiles = MGPU_DIV_UP(count, NV);
    int numBlocks = std::min(context.NumSMs() * 25, numTiles);
    int2 task = mgpu::DivideTaskRange(numTiles, numBlocks);
    
    MGPU_MEM(value_type) reductionDevice = context.Malloc<value_type>(numBlocks + 1);
    	
    // N loads
    mgpu::KernelReduce<Tuning><<<numBlocks, launch.x>>>(data_global, count, task, reductionDevice->get(), op);
    
    // scan the sums to get the carries
    const unsigned int groupsize2 = 256;
    const unsigned int grainsize2 = 3;

    // XXX we could scatter the carries to the output instead of scanning in place
    //     this might simplify the next kernel
    bulk::static_thread_group<groupsize2,grainsize2> group2;
    bulk::async(bulk::par(group2,1), exclusive_scan_n<groupsize2,grainsize2>(), bulk::there, reductionDevice->get(), numBlocks, reductionDevice->get(), 0, thrust::plus<int>());
    
    // do the downsweep - N loads, N stores
    bulk::static_thread_group<groupsize1,grainsize1> group1;
    bulk::async(bulk::par(group1,numBlocks), inclusive_downsweep<groupsize1,grainsize1>(), bulk::there, data_global, count, task, reductionDevice->get(), dest_global, thrust::plus<int>());
  }
}


template<typename InputIterator, typename OutputIterator>
OutputIterator my_inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
{
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);

  ::IncScan<mgpu::MgpuScanTypeInc>(thrust::raw_pointer_cast(&*first),
                                   last - first,
                                   thrust::raw_pointer_cast(&*result),
                                   mgpu::ScanOp<mgpu::ScanOpTypeAdd,int>(),
                                   *ctx);

  return result + (last - first);
}


void my_scan(thrust::device_vector<T> *data)
{
  my_inclusive_scan(data->begin(), data->end(), data->begin());
}


void do_it(size_t n)
{
  thrust::host_vector<T> h_input(n);
  thrust::fill(h_input.begin(), h_input.end(), 1);

  thrust::host_vector<T> h_result(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_result.begin());

  thrust::device_vector<T> d_input = h_input;
  thrust::device_vector<T> d_result(d_input.size());

  my_inclusive_scan(d_input.begin(), d_input.end(), d_result.begin());

  cudaError_t error = cudaDeviceSynchronize();

  if(error)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }

  assert(h_result == d_result);
}


template<typename InputIterator, typename OutputIterator>
OutputIterator mgpu_inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
{
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);

  mgpu::Scan<mgpu::MgpuScanTypeInc>(thrust::raw_pointer_cast(&*first),
                                    last - first,
                                    thrust::raw_pointer_cast(&*result),
                                    mgpu::ScanOp<mgpu::ScanOpTypeAdd,int>(),
                                    (int*)0,
                                    false,
                                    *ctx);

  return result + (last - first);
}


void sean_scan(thrust::device_vector<T> *data)
{
  mgpu_inclusive_scan(data->begin(), data->end(), data->begin());
}


int main()
{
  for(size_t n = 1; n <= 1 << 20; n <<= 1)
  {
    std::cout << "Testing n = " << n << std::endl;
    do_it(n);
  }

  thrust::default_random_engine rng;
  for(int i = 0; i < 20; ++i)
  {
    size_t n = rng() % (1 << 20);
   
    std::cout << "Testing n = " << n << std::endl;
    do_it(n);
  }

  thrust::device_vector<T> vec(1 << 28);

  sean_scan(&vec);
  double sean_msecs = time_invocation_cuda(50, sean_scan, &vec);

  my_scan(&vec);
  double my_msecs = time_invocation_cuda(50, my_scan, &vec);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "My time: " << my_msecs << " ms" << std::endl;

  std::cout << "My relative performance: " << sean_msecs / my_msecs << std::endl;

  return 0;
}

