#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/detail/swap.h>
#include <bulk/bulk.hpp>
#include "time_invocation_cuda.hpp"


template<int NT, int VT, typename InputIt, typename T>
__device__
void DeviceGather(int count, InputIt data, int indices[VT], int tid, T* reg, bool sync)
{
  if(count >= NT * VT)
  {
    #pragma unroll
    for(int i = 0; i < VT; ++i)
    {
      reg[i] = data[indices[i]];
    }
  }
  else
  {
    #pragma unroll
    for(int i = 0; i < VT; ++i)
    {
      int index = NT * i + tid;
      if(index < count)
      {
        reg[i] = data[indices[i]];
      }
    }
  }
  if(sync) __syncthreads();
}


template<int NT, int grainsize, typename KeyType, typename ValType, typename Comp>
__device__
void CTAMergesort(KeyType threadKeys[grainsize], ValType threadValues[grainsize], KeyType* keys_shared, ValType* values_shared, int count, int tid, Comp comp)
{
  bulk::agent<grainsize> exec(threadIdx.x);

  // each agent sorts its local partition
  int local_offset = grainsize * tid;
  int local_size = max(0, min(grainsize, count - local_offset));
  bulk::stable_sort_by_key(bulk::bound<grainsize>(exec), threadKeys, threadKeys + local_size, threadValues, comp);
  
  // store the sorted partitions back to smem
  bulk::copy_n(bulk::bound<grainsize>(exec), threadKeys, local_size, keys_shared + local_offset);
  
  // iteratively merge lists until the entire CTA is sorted.
  for(int coop = 2; coop <= NT; coop *= 2)
  {
    int indices[grainsize];
    mgpu::CTABlocksortPass<NT, grainsize>(keys_shared, tid, count, coop, threadKeys, indices, comp);
    
    // Exchange the values through shared memory.
    bulk::copy_n(bulk::bound<grainsize>(exec), threadValues, local_size, values_shared + local_offset);

    ::DeviceGather<NT, grainsize>(NT * grainsize, values_shared, indices, tid, threadValues, true);
    
    // Store results in shared memory in sorted order.
    bulk::copy_n(bulk::bound<grainsize>(exec), threadKeys, local_size, keys_shared + local_offset);

    __syncthreads();
  }
}


template<typename Tuning, typename KeyIt1, typename KeyIt2, typename ValIt1, typename ValIt2, typename Comp>
__global__ void KernelBlocksort(KeyIt1 keysSource_global, ValIt1 valsSource_global, int count, KeyIt2 keysDest_global, ValIt2 valsDest_global, Comp comp)
{
  typedef MGPU_LAUNCH_PARAMS Params;
  typedef typename std::iterator_traits<KeyIt1>::value_type KeyType;
  typedef typename std::iterator_traits<ValIt1>::value_type ValType;
  
  const int groupsize = Params::NT;
  const int grainsize = Params::VT;
  const int tile_size = groupsize * grainsize;
  union Shared
  {
    KeyType keys[groupsize * (grainsize + 1)];
    ValType values[tile_size];
  };
  __shared__ Shared shared;
  
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int gid = tile_size * block;
  int count2 = min(tile_size, count - gid);
  
  // Load the values into thread order.
  ValType threadValues[grainsize];
  mgpu::DeviceGlobalToShared<groupsize, grainsize>(count2, valsSource_global + gid, tid, shared.values);
  mgpu::DeviceSharedToThread<grainsize>(shared.values, tid, threadValues);
  
  // Load keys into shared memory and transpose into register in thread order.
  KeyType threadKeys[grainsize];
  mgpu::DeviceGlobalToShared<groupsize, grainsize>(count2, keysSource_global + gid, tid, shared.keys);
  mgpu::DeviceSharedToThread<grainsize>(shared.keys, tid, threadKeys);
  
  ::CTAMergesort<groupsize, grainsize, true>(threadKeys, threadValues, shared.keys, shared.values, count2, tid, comp);
  
  // Store the sorted keys to global.
  mgpu::DeviceSharedToGlobal<groupsize, grainsize>(count2, shared.keys, tid, keysDest_global + gid);
  mgpu::DeviceThreadToShared<grainsize>(threadValues, tid, shared.values);
  mgpu::DeviceSharedToGlobal<groupsize, grainsize>(count2, shared.values, tid, valsDest_global + gid);
}


template<typename KeyType, typename ValType, typename Comp>
void MergesortPairs(KeyType* keys_global, ValType* values_global, int count, Comp comp, mgpu::CudaContext& context)
{
  const int NT = 256;
  const int VT = 11;
  typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
  int2 launch = Tuning::GetLaunchParams(context);
  
  const int NV = launch.x * launch.y;
  int numBlocks = MGPU_DIV_UP(count, NV);
  int numPasses = mgpu::FindLog2(numBlocks, true);
  
  MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
  MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(count);
  KeyType* keysSource = keys_global;
  KeyType* keysDest = keysDestDevice->get();
  ValType* valsSource = values_global;
  ValType* valsDest = valsDestDevice->get();
  
  mgpu::KernelBlocksort<Tuning, true><<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, valsSource, count, (1 & numPasses) ? keysDest : keysSource, (1 & numPasses) ? valsDest : valsSource, comp);

  if(1 & numPasses)
  {
    std::swap(keysSource, keysDest);
    std::swap(valsSource, valsDest);
  }
  
  for(int pass = 0; pass < numPasses; ++pass) 
  {
    int coop = 2<< pass;
    MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions<mgpu::MgpuBoundsLower>(keysSource, count, keysSource, 0, NV, coop, comp, context);
    
    mgpu::KernelMerge<Tuning, true, true><<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, valsSource, count, keysSource, valsSource, 0, partitionsDevice->get(), coop, keysDest, valsDest, comp);

    std::swap(keysDest, keysSource);
    std::swap(valsDest, valsSource);
  }
}


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void my_sort_by_key_(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, Compare comp)
{
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);
  ::MergesortPairs(thrust::raw_pointer_cast(&*keys_first),
                   thrust::raw_pointer_cast(&*values_first),
                   keys_last - keys_first,
                   comp,
                   *ctx);
}


struct my_less
{
  template<typename T>
  __host__ __device__
  bool operator()(const T &x, const T& y)
  {
    return x < y;
  }
};


template<typename T>
void my_sort_by_key(const thrust::device_vector<T> *unsorted_keys,
                    const thrust::device_vector<T> *unsorted_values,
                    thrust::device_vector<T> *sorted_keys,
                    thrust::device_vector<T> *sorted_values)
{
  *sorted_keys = *unsorted_keys;
  *sorted_values = *unsorted_values;
  my_sort_by_key_(sorted_keys->begin(), sorted_keys->end(), sorted_values->begin(), my_less());
}


template<typename T>
void sean_sort_by_key(const thrust::device_vector<T> *unsorted_keys,
                      const thrust::device_vector<T> *unsorted_values,                    
                      thrust::device_vector<T> *sorted_keys,
                      thrust::device_vector<T> *sorted_values)
{
  *sorted_keys = *unsorted_keys;
  *sorted_values = *unsorted_values;
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);
  mgpu::MergesortPairs(thrust::raw_pointer_cast(sorted_keys->data()),
                       thrust::raw_pointer_cast(sorted_values->data()),
                       sorted_keys->size(),
                       my_less(),
                       *ctx);
}


template<typename T>
void thrust_sort_by_key(const thrust::device_vector<T> *unsorted_keys,
                        const thrust::device_vector<T> *unsorted_values,
                        thrust::device_vector<T> *sorted_keys,
                        thrust::device_vector<T> *sorted_values)
{
  *sorted_keys = *unsorted_keys;
  *sorted_values = *unsorted_values;
  thrust::sort_by_key(sorted_keys->begin(), sorted_keys->end(), sorted_values->begin(), my_less());
}


template<typename T>
struct hash
{
  template<typename Integer>
  __device__ __device__
  T operator()(Integer x)
  {
    x = (x+0x7ed55d16) + (x<<12);
    x = (x^0xc761c23c) ^ (x>>19);
    x = (x+0x165667b1) + (x<<5);
    x = (x+0xd3a2646c) ^ (x<<9);
    x = (x+0xfd7046c5) + (x<<3);
    x = (x^0xb55a4f09) ^ (x>>16);
    return x;
  }
};


template<typename Vector>
void random_fill(Vector &vec)
{
  thrust::tabulate(vec.begin(), vec.end(), hash<typename Vector::value_type>());
}


template<typename T>
void compare(size_t n)
{
  thrust::device_vector<T> unsorted_keys(n), unsorted_values(n), sorted_keys(n), sorted_values(n);

  random_fill(unsorted_keys);
  random_fill(unsorted_values);

  my_sort_by_key(&unsorted_keys, &unsorted_values, &sorted_keys, &sorted_values);
  double my_msecs = time_invocation_cuda(20, my_sort_by_key<T>, &unsorted_keys, &unsorted_values, &sorted_keys, &sorted_values);

  sean_sort_by_key(&unsorted_keys, &unsorted_values, &sorted_keys, &sorted_values);
  double sean_msecs = time_invocation_cuda(20, sean_sort_by_key<T>, &unsorted_keys, &unsorted_values, &sorted_keys, &sorted_values);

  thrust_sort_by_key(&unsorted_keys, &unsorted_values, &sorted_keys, &sorted_values);
  double thrust_msecs = time_invocation_cuda(20, thrust_sort_by_key<T>, &unsorted_keys, &unsorted_values, &sorted_keys, &sorted_values);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;

  std::cout << "Performance relative to Sean: " << sean_msecs / my_msecs << std::endl;
  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;
}


template<typename T>
void validate(size_t n)
{
  thrust::device_vector<T> unsorted_keys(n), unsorted_values(n);

  random_fill(unsorted_keys);
  random_fill(unsorted_values);

  thrust::device_vector<T> ref_keys = unsorted_keys;
  thrust::device_vector<T> ref_values = unsorted_values;
  thrust::sort_by_key(ref_keys.begin(), ref_keys.end(), ref_values.begin(), my_less());

  thrust::device_vector<T> sorted_keys = unsorted_keys;
  thrust::device_vector<T> sorted_values = unsorted_values;

  my_sort_by_key_(sorted_keys.begin(), sorted_keys.end(), sorted_values.begin(), my_less());

  std::cout << "CUDA error: " << cudaGetErrorString(cudaThreadSynchronize()) << std::endl;

  assert(sorted_keys == ref_keys);
  assert(sorted_values == ref_values);
}


int main()
{
  size_t n = 12345678;

  //validate<int>(n);
  validate<double>(n);

  std::cout << "Large input: " << std::endl;
  std::cout << "int: " << std::endl;
  compare<int>(n);

  std::cout << "float: " << std::endl;
  compare<float>(n);

  std::cout << "double: " << std::endl;
  compare<double>(n);
  std::cout << std::endl;

  return 0;
}

