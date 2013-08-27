#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/detail/minmax.h>
#include <bulk/bulk.hpp>
#include "time_invocation_cuda.hpp"
#include "join_iterator.hpp"


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


template<std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename Compare>
__device__
thrust::pair<RandomAccessIterator5,RandomAccessIterator6>
merge_by_key(bulk::bounded<
               groupsize*grainsize,
               bulk::concurrent_group<bulk::agent<grainsize>, groupsize>
             > &g,
             RandomAccessIterator1 keys_first1, RandomAccessIterator1 keys_last1,
             RandomAccessIterator2 keys_first2, RandomAccessIterator2 keys_last2,
             RandomAccessIterator3 values_first1,
             RandomAccessIterator4 values_first2,
             RandomAccessIterator5 keys_result,
             RandomAccessIterator6 values_result,
             Compare comp)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  typedef typename thrust::iterator_value<RandomAccessIterator5>::type key_type;

  // XXX use malloc
  union shared
  {
    key_type  keys[groupsize * grainsize];
    size_type indices[groupsize * grainsize];
  };
  __shared__ shared stage;

  size_type n1 = keys_last1 - keys_first1;
  size_type n2 = keys_last2 - keys_first2;
  size_type  n = n1 + n2;
  
  // copy keys into stage
  bulk::copy_n(g,
               make_join_iterator(keys_first1, n1, keys_first2),
               n,
               stage.keys);

  // find the stage of each agent's sequential merge
  size_type diag = thrust::min<size_type>(n1 + n2, grainsize * g.this_exec.index());
  size_type mp = bulk::merge_path(stage.keys, n1, stage.keys + n1, n2, diag, comp);
  
  // compute the ranges of the sources in the stage.
  size_type start1 = mp;
  size_type start2 = n1 + diag - mp;

  size_type end1 = n1;
  size_type end2 = n1 + n2;
  
  // each agent merges sequentially
  key_type  results[grainsize];
  size_type indices[grainsize];
  bulk::merge_by_key(bulk::bound<grainsize>(g.this_exec),
                     stage.keys + start1, stage.keys + end1,
                     stage.keys + start2, stage.keys + end2,
                     thrust::make_counting_iterator<size_type>(start1),
                     thrust::make_counting_iterator<size_type>(start2),
                     results,
                     indices,
                     comp);
  g.wait();
  
  // each agent stores merged keys back to the stage
  size_type local_offset = grainsize * g.this_exec.index();
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), results, local_size, stage.keys + local_offset);
  g.wait();
  
  // store merged keys to the result
  keys_result = bulk::copy_n(g, stage.keys, n, keys_result);
  
  // each agent copies the indices into the stage
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), indices, local_size, stage.indices + local_offset);
  g.wait();
  
  // gather values into merged order
  values_result = bulk::gather(g,
                               stage.indices, stage.indices + n,
                               make_join_iterator(values_first1, n1, values_first2),
                               values_result);

  return thrust::make_pair(keys_result, values_result);
}


template<typename Tuning, typename KeysIt1, 
	typename KeysIt3, typename ValsIt1,
	typename ValsIt3, typename Comp>
__global__ void my_KernelMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global, int aCount, const int* mp_global, int coop, KeysIt3 keys_global, ValsIt3 vals_global, Comp comp)
{
  typedef int size_type;

  typedef MGPU_LAUNCH_PARAMS Params;
  typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;
  typedef typename std::iterator_traits<ValsIt1>::value_type ValType;
  
  const int NT = Params::NT;
  const int VT = Params::VT;
  
  int block = blockIdx.x;
  
  int4 range = mgpu::ComputeMergeRange(aCount, 0, block, coop, NT * VT, mp_global);

  size_type a0 = range.x;
  size_type a1 = range.y;
  size_type b0 = range.z;
  size_type b1 = range.w;

  bulk::concurrent_group<bulk::agent<VT>, NT> g(0, bulk::agent<VT>(threadIdx.x), blockIdx.x);
  
  merge_by_key<NT, VT>(bulk::bound<NT*VT>(g),
                       aKeys_global + a0, aKeys_global + a1,
                       aKeys_global + b0, aKeys_global + b1,
                       aVals_global + a0,
                       aVals_global + b0,
                       keys_global + NT * VT * block,
                       vals_global + NT * VT * block,
                       comp);
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
    
    my_KernelMerge<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(keysSource, valsSource, count, partitionsDevice->get(), coop, keysDest, valsDest, comp);

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

