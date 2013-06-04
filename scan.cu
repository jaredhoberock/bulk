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
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <bulk/bulk.hpp>


typedef int T;


template<unsigned int grainsize, typename Iterator1, typename Size, typename Iterator2>
__device__
void copy_n_with_grainsize(Iterator1 first, Size n, Iterator2 result)
{
  for(Iterator1 last = first + n;
      first < last;
      first += grainsize, result += grainsize)
  {
    for(int i = 0; i < grainsize; ++i)
    {
      if(i < (last - first))
      {
        result[i] = first[i];
      }
    }
  }
}


template<typename ThreadGroup, typename Iterator, typename Size, typename T, typename BinaryFunction>
__device__ T exclusive_scan_n(ThreadGroup &g, Iterator first, Size n, T init, T *carry_out, BinaryFunction binary_op)
{
  T x;

  int tid = g.this_thread.index();

  if(n > 0 && tid == 0)
  {
    *first = binary_op(init, *first);
  }

  if(tid < n)
  {
    x = first[tid];
  }

  g.wait();

  for(int offset = 1; offset < n; offset += offset)
  {
    if(tid >= offset && tid < n)
    {
      x = binary_op(first[tid - offset], x);
    }

    g.wait();

    if(tid < n)
    {
      first[tid] = x;
    }

    g.wait();
  }

  *carry_out = n > 0 ? first[n - 1] : init;

  if(tid - 1 < n)
  {
    x = tid ? first[tid - 1] : init;
  }

  g.wait();
  
  return x;
}


template<typename ThreadGroup, typename Iterator, typename Size, typename BinaryFunction>
__device__ void inclusive_scan_n(ThreadGroup &g, Iterator first, Size n, BinaryFunction binary_op)
{
  T x;

  int tid = g.this_thread.index();

  if(tid < n)
  {
    x = first[tid];
  }

  g.wait();

  for(int offset = 1; offset < n; offset += offset)
  {
    if(tid >= offset && tid < n)
    {
      x = binary_op(first[tid - offset], x);
    }

    g.wait();

    if(tid < n)
    {
      first[tid] = x;
    }

    g.wait();
  }
}


template<typename ThreadGroup, typename Iterator, typename Size, typename T, typename BinaryFunction>
__device__ T exclusive_scan_n(ThreadGroup &g, Iterator first, Size n, T init, BinaryFunction binary_op)
{
  T x;

  int tid = g.this_thread.index();

  if(n > 0 && tid == 0)
  {
    *first = binary_op(init, *first);
  }

  g.wait();

  if(tid < n)
  {
    x = first[tid];
  }

  g.wait();

  inclusive_scan_n(g, first, n, binary_op);

  T result = n > 0 ? first[n - 1] : init;

  x = (tid == 0 || tid - 1 >= n) ? init : first[tid - 1];

  g.wait();

  if(tid < n)
  {
    first[tid] = x;
  }

  g.wait();

  return result;
}


template<unsigned int size, unsigned int grainsize, bool inclusive>
struct scan
{
  template<typename InputIterator, typename OutputIterator, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<size,grainsize> &this_group, InputIterator cta_global, int count, OutputIterator dest_global, BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<InputIterator>::type input_type;

    // XXX this needs to be inferred from the iterators and binary_op
    typedef typename thrust::iterator_value<OutputIterator>::type intermediate_type;
  
    const unsigned int elements_per_group = size * grainsize;

    // we don't need the inputs and the results at the same time
    // so we can overlay these arrays
    union stage
    {
      input_type *inputs;
      intermediate_type *results;
    };

    stage s_stage;
    s_stage.inputs = reinterpret_cast<input_type*>(bulk::malloc(this_group, elements_per_group * thrust::max<int>(sizeof(input_type), sizeof(intermediate_type))));

    intermediate_type *s_sums = reinterpret_cast<intermediate_type*>(bulk::malloc(this_group, size * sizeof(intermediate_type)));
    
    unsigned int tid = this_group.this_thread.index();
    
    // carry is the sum over all previous iterations
    intermediate_type carry = cta_global[0];
  
    if(this_group.this_thread.index() == 0)
    {
      dest_global[0] = carry;
    }
  
    for(int start = 1; start < count; start += elements_per_group)
    {
      int count2 = min(elements_per_group, count - start);
  
      // stage data through shared memory
      bulk::copy_n(this_group, cta_global + start, count2, s_stage.inputs);
      
      // Transpose data into register in thread order. Reduce terms serially.
      input_type local_inputs[grainsize];
  
      int local_size = max(0,min(grainsize, count2 - grainsize * tid));
  
      int local_offset = grainsize * tid;
  
      intermediate_type x = 0;
  
      if(local_size > 0)
      {
        // XXX would be cool simply to call
        // bulk::copy_n(this_group.this_thread, ...) instead
        copy_n_with_grainsize<grainsize>(s_stage.inputs + local_offset, local_size, local_inputs);
  
        // XXX this should actually be accumulate because we desire non-commutativity
        x = thrust::reduce(thrust::seq, local_inputs + 1, local_inputs + local_size, local_inputs[0], binary_op);
  
        s_sums[tid] = x;
      }
  
      this_group.wait();
  
      // scan this group's sums
      // XXX is this really the correct number of sums?
      //     it should be divide_ri(count2, grainsize)
      carry = ::exclusive_scan_n(this_group, s_sums, min(size,count2), carry, binary_op);
  
      // each thread does an inplace scan locally while incorporating the carries
      if(local_size > 0)
      {
        x = s_sums[tid];
  
        if(inclusive)
        {
          local_inputs[0] = binary_op(x,local_inputs[0]);
  
          // XXX would be cool simply to call
          // bulk::inclusive_scan(this_group.this_thread, ...) instead
          thrust::inclusive_scan(thrust::seq, local_inputs, local_inputs + local_size, local_inputs, binary_op);
        }
        else
        {
          // XXX would be cool simply to call
          // bulk::exclusive_scan(this_group.this_thread, ...) instead
          thrust::exclusive_scan(thrust::seq, local_inputs, local_inputs + local_size, local_inputs, x, binary_op);
        }
  
        // XXX would be cool simply to call
        // bulk::copy_n(this_group.this_thread, ...) instead
        copy_n_with_grainsize<grainsize>(local_inputs, local_size, s_stage.results + local_offset);
      }
  
      this_group.wait();
      
      // store results
      bulk::copy_n(this_group, s_stage.results, count2, dest_global + start);
    }

    bulk::free(this_group, s_stage.inputs);
    bulk::free(this_group, s_sums);
  }
};


template<mgpu::MgpuScanType Type, typename InputIt, typename OutputIt, typename Op>
void IncScan(InputIt data_global, int count, OutputIt dest_global, Op op, mgpu::CudaContext& context)
{
  typedef typename Op::value_type value_type;
  typedef typename Op::result_type result_type;
  
  const int CutOff = 20000;

  if(count < CutOff)
  {
    const int NT = 512;
    const int VT = 3;

    bulk::static_thread_group<NT,VT> group;
    bulk::async(bulk::par(group, 1), scan<NT,VT,true>(), bulk::there, data_global, count, dest_global, thrust::plus<int>());
  }
  else
  {
    // Run the parallel raking reduce as an upsweep.
    const int NT = 128;
    const int VT = 7;
    typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
    int2 launch = Tuning::GetLaunchParams(context);
    const int NV = launch.x * launch.y;
    
    int numTiles = MGPU_DIV_UP(count, NV);
    int numBlocks = std::min(context.NumSMs() * 25, numTiles);
    int2 task = mgpu::DivideTaskRange(numTiles, numBlocks);
    
    MGPU_MEM(value_type) reductionDevice = context.Malloc<value_type>(numBlocks + 1);
    	
    mgpu::KernelReduce<Tuning><<<numBlocks, launch.x>>>(data_global, count, task, reductionDevice->get(), op);
    
    // Run a parallel latency-oriented scan to reduce the spine of the 
    // raking reduction.
    const int NT2 = 256;
    const int VT2 = 3;

    bulk::static_thread_group<NT2,VT2> group;
    bulk::async(bulk::par(group,1), scan<NT2,VT2,false>(), bulk::there, reductionDevice->get(), numBlocks, reductionDevice->get(), thrust::plus<int>());
    
    // Run a raking scan as a downsweep.
    mgpu::KernelScanDownsweep<Tuning, Type><<<numBlocks, launch.x>>>(data_global, count, task, reductionDevice->get(), dest_global, false, op);
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

  thrust::device_vector<T> vec(1 << 24);

  sean_scan(&vec);
  double sean_msecs = time_invocation_cuda(50, sean_scan, &vec);

  my_scan(&vec);
  double my_msecs = time_invocation_cuda(50, my_scan, &vec);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "My time: " << my_msecs << " ms" << std::endl;

  std::cout << "My relative performance: " << sean_msecs / my_msecs << std::endl;

  return 0;
}

