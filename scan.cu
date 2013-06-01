#include <moderngpu.cuh>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cassert>
#include <iostream>
#include "time_invocation_cuda.hpp"
#include <thrust/detail/temporary_array.h>
#include <bulk/thread_group.hpp>
#include <bulk/algorithm.hpp>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>


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


// Scan inputs on a single CTA. Optionally output the total to dest_global at
// totalIndex.
template<int NT, int VT, mgpu::MgpuScanType Type, typename InputIt, typename OutputIt, typename Op>
__global__ void my_KernelParallelScan(InputIt cta_global, int count, Op op, OutputIt dest_global)
{
  bulk::static_thread_group<NT,VT> this_group;

  typedef typename Op::input_type input_type;
  typedef typename Op::value_type value_type;
  typedef typename Op::result_type result_type;
  const int NV = NT * VT;
  
  typedef mgpu::CTAScan<NT, Op> S;
  union Shared
  {
    typename S::Storage scan;
    input_type inputs[NV];
    result_type results[NV];
  };
  __shared__ Shared shared;
  
  int tid = threadIdx.x;
  
  // carry is the sum over all previous iterations
  value_type carry = cta_global[0];

  for(int start = 1; start < count; start += NV)
  {
    int count2 = min(NV, count - start);

    // copy data into shared memory
    bulk::copy_n(this_group, cta_global + start, count2, shared.inputs);
    
    // Transpose data into register in thread order. Reduce terms serially.
    input_type local_inputs[VT];

    int local_size = max(0,min(VT, count2 - VT * tid));

    value_type x = 0;
    if(local_size > 0)
    {
      int src_offset = VT * tid;

      // XXX would be cool simply to call
      // bulk::copy_n(this_group.this_thread, ...) instead
      copy_n_with_grainsize<VT>(shared.inputs + src_offset, local_size, local_inputs);

      // XXX this should actually be accumulate because we desire non-commutativity
      x = thrust::reduce(thrust::seq, local_inputs + 1, local_inputs + local_size, local_inputs[0]);
    }

    this_group.wait();
    		
    // Scan the reduced terms.
    // XXX we should pass in the carry here as the init since this is an exclusive_scan
    value_type pass_carry;
    x = S::Scan(tid, x, shared.scan, &pass_carry, mgpu::MgpuScanTypeExc, op);

    x = op.Plus(carry, x);
    carry = op.Plus(carry, pass_carry);

    //// now we do an inplace scan while incorporating the carries
    //if(local_size > 0)
    //{
    //  if(Type == mgpu::MgpuScanTypeInc)
    //  {
    //    local_inputs[0] = op.Plus(x, local_inputs[0]);
    //    thrust::inclusive_scan(thrust::seq, local_inputs, local_inputs + local_size, local_inputs);
    //  }
    //  else
    //  {
    //    thrust::exclusive_scan(thrust::seq, local_inputs, local_inputs + local_size, local_inputs, x);
    //  }

    //  copy_n_with_grainsize<VT>(local_inputs, local_size, shared.results);
    //}
    
    #pragma unroll
    for(int i = 0; i < VT; ++i)
    {
      int index = VT * tid + i;
      if(index < count2)
      {
        value_type x2 = op.Plus(x, local_inputs[i]);
      
        // For inclusive scan, set the new value then store.
        // For exclusive scan, store the old value then set the new one.
        if(mgpu::MgpuScanTypeInc == Type) x = x2;
        shared.results[index] = op.Combine(local_inputs[i], x);
        if(mgpu::MgpuScanTypeExc == Type) x = x2;
      }
    }

    this_group.wait();
    
    // store results
    bulk::copy_n(this_group, shared.results, count2, dest_global + start);
  }
}


template<mgpu::MgpuScanType Type, typename InputIt, typename OutputIt, typename Op>
void Scan(InputIt data_global, int count, OutputIt dest_global, Op op, mgpu::CudaContext& context)
{
  typedef typename Op::value_type value_type;
  typedef typename Op::result_type result_type;
  
  const int CutOff = 20000;

  if(count < CutOff)
  {
    const int NT = 512;
    const int VT = 3;
    
    my_KernelParallelScan<NT, VT, Type><<<1, NT>>>(data_global, count, op, dest_global);
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
    my_KernelParallelScan<NT2, VT2, mgpu::MgpuScanTypeExc><<<1, NT2>>>(reductionDevice->get(), numBlocks, mgpu::ScanOpValue<Op>(op), reductionDevice->get());
    
    // Run a raking scan as a downsweep.
    mgpu::KernelScanDownsweep<Tuning, Type><<<numBlocks, launch.x>>>(data_global, count, task, reductionDevice->get(), dest_global, false, op);
  }
}


template<typename InputIterator, typename OutputIterator>
OutputIterator my_inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
{
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);

  ::Scan<mgpu::MgpuScanTypeInc>(thrust::raw_pointer_cast(&*first),
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
  thrust::sequence(h_input.begin(), h_input.end());

  thrust::host_vector<T> h_result(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_result.begin());

  thrust::device_vector<T> d_input = h_input;
  thrust::device_vector<T> d_result(d_input.size());

  my_inclusive_scan(d_input.begin(), d_input.end(), d_result.begin());

  cudaError_t error = cudaGetLastError();

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
  double sean_msecs = time_invocation_cuda(20, sean_scan, &vec);

  my_scan(&vec);
  double my_msecs = time_invocation_cuda(20, my_scan, &vec);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "My time: " << my_msecs << " ms" << std::endl;

  std::cout << "My relative performance: " << sean_msecs / my_msecs << std::endl;

  return 0;
}

