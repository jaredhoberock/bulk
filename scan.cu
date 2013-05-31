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


typedef int T;


// Scan inputs on a single CTA. Optionally output the total to dest_global at
// totalIndex.
template<int NT, int VT, mgpu::MgpuScanType Type, typename InputIt, typename OutputIt, typename Op>
__global__ void my_KernelParallelScan(InputIt cta_global, int count, Op op, typename Op::value_type* total_global, typename Op::result_type* end_global, OutputIt dest_global)
{
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
  
  // total is the sum of encountered elements. It's undefined on the first 
  // loop iteration.
  value_type total = op.Extract(op.Identity(), -1);
  bool totalDefined = false;
  int start = 0;
  while(start < count)
  {
    // Load data into shared memory.
    int count2 = min(NV, count - start);
    mgpu::DeviceGlobalToShared<NT, VT>(count2, cta_global + start, tid, shared.inputs);
    
    // Transpose data into register in thread order. Reduce terms serially.
    input_type inputs[VT];
    value_type values[VT];
    value_type x = op.Extract(op.Identity(), -1);
    #pragma unroll
    for(int i = 0; i < VT; ++i)
    {
      int index = VT * tid + i;
      if(index < count2)
      {
        inputs[i] = shared.inputs[index];
        values[i] = op.Extract(inputs[i], start + index);
        x = i ? op.Plus(x, values[i]) : values[i];
      }
    }
    __syncthreads();
    		
    // Scan the reduced terms.
    value_type passTotal;
    x = S::Scan(tid, x, shared.scan, &passTotal, mgpu::MgpuScanTypeExc, op);
    if(totalDefined)
    {
      x = op.Plus(total, x);
      total = op.Plus(total, passTotal);
    }
    else
    {
      total = passTotal;
    }
    
    #pragma unroll
    for(int i = 0; i < VT; ++i)
    {
      int index = VT * tid + i;
      if(index < count2)
      {
        // If this is not the first element in the scan, add x values[i]
        // into x. Otherwise initialize x to values[i].
        value_type x2 = (i || tid || totalDefined) ? op.Plus(x, values[i]) : values[i];
      
        // For inclusive scan, set the new value then store.
        // For exclusive scan, store the old value then set the new one.
        if(mgpu::MgpuScanTypeInc == Type) x = x2;
        shared.results[index] = op.Combine(inputs[i], x);
        if(mgpu::MgpuScanTypeExc == Type) x = x2;
      }
    }
    __syncthreads();
    
    mgpu::DeviceSharedToGlobal<NT, VT>(count2, shared.results, tid, dest_global + start);

    start += NV;
    totalDefined = true;
  }
  
  if(total_global && !tid)
  {
    *total_global = total;
  }
  
  if(end_global && !tid)
  {
    *end_global = op.Combine(op.Identity(), total);
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
    
    my_KernelParallelScan<NT, VT, Type><<<1, NT>>>(data_global, count, op, (value_type*)0, (result_type*)0, dest_global);
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

    value_type* totalDevice = (value_type*)0;
    	
    mgpu::KernelReduce<Tuning><<<numBlocks, launch.x>>>(data_global, count, task, reductionDevice->get(), op);
    
    // Run a parallel latency-oriented scan to reduce the spine of the 
    // raking reduction.
    const int NT2 = 256;
    const int VT2 = 3;
    mgpu::KernelParallelScan<NT2, VT2, mgpu::MgpuScanTypeExc><<<1, NT2>>>(reductionDevice->get(), numBlocks, mgpu::ScanOpValue<Op>(op), totalDevice, (value_type*)0, reductionDevice->get());
    
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

