#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/detail/minmax.h>
#include <bulk/bulk.hpp>
#include "time_invocation_cuda.hpp"


template<typename Tuning, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
__global__ void stable_sort_each_kernel(RandomAccessIterator1 keys_first, RandomAccessIterator2 values_first, int count, Compare comp)
{
  typedef MGPU_LAUNCH_PARAMS Params;
  
  const int groupsize = Params::NT;
  const int grainsize = Params::VT;
  const int tilesize = groupsize * grainsize;
  
  bulk::concurrent_group<bulk::agent<grainsize>,groupsize> g(0, bulk::agent<grainsize>(threadIdx.x), blockIdx.x);

  int gid = tilesize * g.index();
  int count2 = min(tilesize, count - gid);

  bulk::stable_sort_by_key(bulk::bound<tilesize>(g), keys_first + gid, keys_first + gid + count2, values_first + gid, comp);
}


template<int NT, int VT, typename It1, typename It2, typename T, typename Comp>
__device__
void my_DeviceMergeKeysIndices(It1 a_global, It2 b_global, int4 range, int tid, T* keys_shared, T* results, int* indices, Comp comp)
{
  int a0 = range.x;
  int a1 = range.y;
  int b0 = range.z;
  int b1 = range.w;
  int aCount = a1 - a0;
  int bCount = b1 - b0;
  
  // Load the data into shared memory.
  mgpu::DeviceLoad2ToShared<NT, VT, VT>(a_global + a0, aCount, b_global + b0, bCount, tid, keys_shared);
  
  // Run a merge path to find the start of the serial merge for each thread.
  int diag = VT * tid;
  int mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(keys_shared, aCount, keys_shared + aCount, bCount, diag, comp);
  
  // Compute the ranges of the sources in shared memory.
  int a0tid = mp;
  int a1tid = aCount;
  int b0tid = aCount + diag - mp;
  int b1tid = aCount + bCount;
  
  // Serial merge into register.
  mgpu::SerialMerge<VT, true>(keys_shared, a0tid, a1tid, b0tid, b1tid, results, indices, comp);
}


template<int NT, int VT, typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2, typename KeyType, typename ValsIt3, typename Comp>
__device__
void my_DeviceMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
                    KeysIt2 bKeys_global, ValsIt2 bVals_global, int tid, int block, int4 range,
	            KeyType* keys_shared, int* indices_shared, KeysIt3 keys_global,
	            ValsIt3 vals_global, Comp comp)
{
  KeyType results[VT];
  int indices[VT];
  my_DeviceMergeKeysIndices<NT, VT>(aKeys_global, bKeys_global, range, tid, keys_shared, results, indices, comp);
  
  // Store merge results back to shared memory.
  mgpu::DeviceThreadToShared<VT>(results, tid, keys_shared);
  
  // Store merged keys to global memory.
  int aCount = range.y - range.x;
  int bCount = range.w - range.z;
  mgpu::DeviceSharedToGlobal<NT, VT>(aCount + bCount, keys_shared, tid, keys_global + NT * VT * block);
  
  // Copy the values.
  mgpu::DeviceThreadToShared<VT>(indices, tid, indices_shared);
  
  mgpu::DeviceTransferMergeValues<NT, VT>(aCount + bCount, aVals_global + range.x, bVals_global + range.z, aCount, indices_shared, tid, vals_global + NT * VT * block);
}


template<typename Tuning, typename KeysIt1, 
	typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
	typename ValsIt3, typename Comp>
__global__ void my_KernelMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global, int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount, const int* mp_global, int coop, KeysIt3 keys_global, ValsIt3 vals_global, Comp comp)
{
  typedef MGPU_LAUNCH_PARAMS Params;
  typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;
  typedef typename std::iterator_traits<ValsIt1>::value_type ValType;
  
  const int NT = Params::NT;
  const int VT = Params::VT;
  const int NV = NT * VT;
  union Shared
  {
    KeyType keys[NT * (VT + 1)];
    int indices[NV];
  };
  __shared__ Shared shared;
  
  int tid = threadIdx.x;
  int block = blockIdx.x;
  
  int4 range = mgpu::ComputeMergeRange(aCount, bCount, block, coop, NT * VT, mp_global);
  
  my_DeviceMerge<NT, VT>(aKeys_global, aVals_global, bKeys_global, bVals_global, tid, block, range, shared.keys, shared.indices, keys_global, vals_global, comp);
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

  stable_sort_each_kernel<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, valsSource, count, comp);
  
  for(int pass = 0; pass < numPasses; ++pass) 
  {
    int coop = 2<< pass;
    MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions<mgpu::MgpuBoundsLower>(keysSource, count, keysSource, 0, NV, coop, comp, context);
    
    my_KernelMerge<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, valsSource, count, keysSource, valsSource, 0, partitionsDevice->get(), coop, keysDest, valsDest, comp);

    std::swap(keysDest, keysSource);
    std::swap(valsDest, valsSource);
  }

  if(1 & numPasses)
  {
    thrust::copy_n(thrust::cuda::tag(), thrust::make_zip_iterator(thrust::make_tuple(keysSource, valsSource)), count, thrust::make_zip_iterator(thrust::make_tuple(keysDest, valsDest)));
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

