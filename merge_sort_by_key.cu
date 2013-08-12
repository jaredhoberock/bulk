#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/detail/minmax.h>
#include <bulk/bulk.hpp>
#include "time_invocation_cuda.hpp"


template<std::size_t groupsize, std::size_t grainsize, typename KeyType, typename ValType, typename Comp>
__device__
void inplace_merge_adjacent_partitions(KeyType threadKeys[grainsize], ValType threadValues[grainsize], void* stage_ptr, int count, int local_size, Comp comp)
{
  union stage_t
  {
    KeyType *keys;
    ValType *vals;
  };
  
  stage_t stage;
  stage.keys = reinterpret_cast<KeyType*>(stage_ptr);

  bulk::agent<grainsize> exec(threadIdx.x);

  typedef typename bulk::agent<grainsize>::size_type size_type;

  size_type local_offset = grainsize * exec.index();

  for(size_type num_agents_per_merge = 2; num_agents_per_merge <= groupsize; num_agents_per_merge *= 2)
  {
    // copy keys into the stage so we can dynamically index them
    bulk::copy_n(bulk::bound<grainsize>(exec), threadKeys, local_size, stage.keys + local_offset);

    __syncthreads();

    // find the index of the first array this agent will merge
    size_type list = ~(num_agents_per_merge - 1) & exec.index();
    size_type diag = thrust::min<size_type>(count, grainsize * ((num_agents_per_merge - 1) & exec.index()));
    size_type start = grainsize * list;

    // the size of each of the two input arrays we're merging
    size_type input_size = grainsize * (num_agents_per_merge / 2);

    size_type partition_first1 = thrust::min<size_type>(count, start);
    size_type partition_first2 = thrust::min<size_type>(count, partition_first1 + input_size);
    size_type partition_last2  = thrust::min<size_type>(count, partition_first2 + input_size);

    size_type n1 = partition_first2 - partition_first1;
    size_type n2 = partition_last2  - partition_first2;

    size_type mp = bulk::merge_path(stage.keys + partition_first1, n1, stage.keys + partition_first2, n2, diag, comp);

    // each agent merges sequentially locally
    // note the source index of each merged value so that we can gather values into merged order later
    size_type gather_indices[grainsize];
    bulk::merge_by_key(bulk::bound<grainsize>(exec),
                       stage.keys + partition_first1 + mp,        stage.keys + partition_first2,
                       stage.keys + partition_first2 + diag - mp, stage.keys + partition_last2,
                       thrust::make_counting_iterator<size_type>(partition_first1 + mp),
                       thrust::make_counting_iterator<size_type>(partition_first2 + diag - mp),
                       threadKeys,
                       gather_indices,
                       comp);
    
    // move values into the stage so we can index them
    bulk::copy_n(bulk::bound<grainsize>(exec), threadValues, local_size, stage.vals + local_offset);

    // gather values into registers
    bulk::gather(bulk::bound<grainsize>(exec), gather_indices, gather_indices + local_size, stage.vals, threadValues);

    __syncthreads();
  } // end for
} // end inplace_merge_adjacent_partitions()


template<std::size_t groupsize, std::size_t grainsize, typename KeyType, typename ValType, typename Comp>
__device__
void my_CTAMergesort(KeyType threadKeys[grainsize], ValType threadValues[grainsize], void* stage_ptr, int count, Comp comp)
{
  bulk::agent<grainsize> exec(threadIdx.x);
  typedef typename bulk::agent<grainsize>::size_type size_type;

  size_type local_offset = grainsize * exec.index();
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, count - local_offset));

  bulk::stable_sort_by_key(bulk::bound<grainsize>(exec), threadKeys, threadKeys + local_size, threadValues, comp);
  
  // Recursively merge lists until the entire CTA is sorted.
  // avoid dynamic sizes when possible
  if(count == groupsize * grainsize)
  {
    inplace_merge_adjacent_partitions<groupsize, grainsize>(threadKeys, threadValues, stage_ptr, groupsize * grainsize, grainsize, comp);
  } // end if
  else
  {
    inplace_merge_adjacent_partitions<groupsize, grainsize>(threadKeys, threadValues, stage_ptr, count, local_size, comp);
  } // end else
} // end my_CTAMergesort()


template<typename Tuning, typename KeyIt1, typename KeyIt2, typename ValIt1, typename ValIt2, typename Comp>
__global__ void my_KernelBlocksort(KeyIt1 keysSource_global, ValIt1 valsSource_global, int count, KeyIt2 keysDest_global, ValIt2 valsDest_global, Comp comp)
{
  typedef MGPU_LAUNCH_PARAMS Params;
  typedef typename std::iterator_traits<KeyIt1>::value_type KeyType;
  typedef typename std::iterator_traits<ValIt1>::value_type ValType;
  
  const int NT = Params::NT;
  const int VT = Params::VT;
  const int NV = NT * VT;
  union Shared 
  {
    KeyType keys[NV];
    ValType values[NV];
  };
  __shared__ Shared shared;
  
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int gid = NV * block;
  int count2 = min(NV, count - gid);
  
  // Load the values into thread order.
  ValType threadValues[VT];
  mgpu::DeviceGlobalToShared<NT, VT>(count2, valsSource_global + gid, tid, shared.values);
  mgpu::DeviceSharedToThread<VT>(shared.values, tid, threadValues);
  
  // Load keys into shared memory and transpose into register in thread order.
  KeyType threadKeys[VT];
  mgpu::DeviceGlobalToShared<NT, VT>(count2, keysSource_global + gid, tid, shared.keys);
  mgpu::DeviceSharedToThread<VT>(shared.keys, tid, threadKeys);

  my_CTAMergesort<NT, VT>(threadKeys, threadValues, shared.keys, count2, comp);
  
  // Store the sorted keys to global.
  mgpu::DeviceThreadToShared<VT>(threadKeys, tid, shared.keys);
  mgpu::DeviceSharedToGlobal<NT, VT>(count2, shared.keys, tid, keysDest_global + gid);
  
  mgpu::DeviceThreadToShared<VT>(threadValues, tid, shared.values);
  mgpu::DeviceSharedToGlobal<NT, VT>(count2, shared.values, tid, valsDest_global + gid);
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
  
  my_KernelBlocksort<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, valsSource, count, (1 & numPasses) ? keysDest : keysSource, (1 & numPasses) ? valsDest : valsSource, comp);

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

  if(n < 30 && sorted_keys != ref_keys)
  {
    std::cerr << "reference: " << std::endl;

    for(int i = 0; i < n; ++i)
    {
      std::cerr << ref_keys[i] << " ";
    }

    std::cerr << std::endl;


    std::cerr << "output: " << std::endl;

    for(int i = 0; i < n; ++i)
    {
      std::cerr << sorted_keys[i] << " ";
    }

    std::cerr << std::endl;
  }

  assert(sorted_keys == ref_keys);
  assert(sorted_values == ref_values);
}


int main()
{
//  std::cout << "small input: " << std::endl;
//  std::cout << "int: " << std::endl;
//
//  validate<int>(20);

  size_t n = 12345678;

  validate<int>(n);

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

