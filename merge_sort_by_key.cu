#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/detail/minmax.h>
#include <bulk/bulk.hpp>
#include "time_invocation_cuda.hpp"
#include "join_iterator.hpp"


struct stable_sort_each_kernel
{
  template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
  __device__ void operator()(bulk::concurrent_group<bulk::agent<grainsize>, groupsize> &g, RandomAccessIterator1 keys_first, RandomAccessIterator2 values_first, int count, Compare comp)
  {
    typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;
    const size_type tilesize = groupsize * grainsize;
  
    size_type gid = tilesize * g.index();
    size_type count2 = thrust::min<size_type>(tilesize, count - gid);
  
    bulk::stable_sort_by_key(bulk::bound<tilesize>(g), keys_first + gid, keys_first + gid + count2, values_first + gid, comp);
  }
};


template<typename Size>
__device__
thrust::tuple<Size,Size,Size,Size>
  locate_merge_partitions(Size n, Size group_idx, Size num_groups_per_merge, Size num_elements_per_group, Size mp, Size right_mp)
{
  Size first_group_in_partition = ~(num_groups_per_merge - 1) & group_idx;
  Size partition_size = num_elements_per_group * (num_groups_per_merge >> 1);

  Size partition_first1 = num_elements_per_group * first_group_in_partition;
  Size partition_first2 = partition_first1 + partition_size;

  // Locate diag from the start of the A sublist.
  Size diag = num_elements_per_group * group_idx - partition_first1;
  Size start1 = partition_first1 + mp;
  Size end1 = thrust::min<Size>(n, partition_first1 + right_mp);
  Size start2 = thrust::min<Size>(n, partition_first2 + diag - mp);
  Size end2 = thrust::min<Size>(n, partition_first2 + diag + num_elements_per_group - right_mp);
  
  // The end partition of the last group for each merge operation is computed
  // and stored as the begin partition for the subsequent merge. i.e. it is
  // the same partition but in the wrong coordinate system, so its 0 when it
  // should be listSize. Correct that by checking if this is the last group
  // in this merge operation.
  if(num_groups_per_merge - 1 == ((num_groups_per_merge - 1) & group_idx))
  {
    end1 = thrust::min<Size>(n, partition_first1 + partition_size);
    end2 = thrust::min<Size>(n, partition_first2 + partition_size);
  }

  return thrust::make_tuple(start1, end1, start2, end2);
}


struct merge_by_key_kernel
{
  template<std::size_t groupsize,
           std::size_t grainsize,
           typename RandomAccessIterator1, 
	   typename RandomAccessIterator2,
           typename RandomAccessIterator3,
	   typename RandomAccessIterator4,
           typename Compare>
  __device__ void operator()(bulk::concurrent_group<bulk::agent<grainsize>, groupsize> &g, RandomAccessIterator1 keys_first, RandomAccessIterator2 values_first, int n, const int *merge_paths, int num_groups_per_merge, RandomAccessIterator3 keys_result, RandomAccessIterator4 values_result, Compare comp)
  {
    typedef typename bulk::concurrent_group<bulk::agent<grainsize>, groupsize>::size_type size_type;

    size_type a0, a1, b0, b1;
    thrust::tie(a0, a1, b0, b1) = locate_merge_partitions<size_type>(n, g.index(), num_groups_per_merge, groupsize * grainsize, merge_paths[g.index()], merge_paths[g.index()+1]);
    
    bulk::merge_by_key(bulk::bound<groupsize*grainsize>(g),
                       keys_first + a0, keys_first + a1,
                       keys_first + b0, keys_first + b1,
                       values_first + a0,
                       values_first + b0,
                       keys_result   + groupsize * grainsize * g.index(),
                       values_result + groupsize * grainsize * g.index(),
                       comp);
  }
};


template<typename KeyType, typename ValType, typename Comp>
void MergesortPairs(KeyType* keys_global, ValType* values_global, int n, Comp comp, mgpu::CudaContext& context)
{
  const int groupsize = 256;
  const int grainsize = 11;
  
  const int tilesize = groupsize * grainsize;
  int num_groups = (n + tilesize - 1) / tilesize;
  int num_passes = mgpu::FindLog2(num_groups, true);
  
  MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(n);
  MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(n);

  KeyType* keysSource = keys_global;
  KeyType* keysDest = keysDestDevice->get();
  ValType* valsSource = values_global;
  ValType* valsDest = valsDestDevice->get();

  int heap_size = tilesize * thrust::max(sizeof(KeyType), sizeof(ValType));

  bulk::async(bulk::grid<groupsize,grainsize>(num_groups, heap_size), stable_sort_each_kernel(), bulk::root.this_exec, keysSource, valsSource, n, comp);
  
  for(int pass = 0; pass < num_passes; ++pass) 
  {
    int num_groups_per_merge = 2 << pass;
    MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions<mgpu::MgpuBoundsLower>(keysSource, n, keysSource, 0, tilesize, num_groups_per_merge, comp, context);
    
    int heap_size = tilesize * thrust::max(sizeof(KeyType), sizeof(int));
    bulk::async(bulk::grid<groupsize,grainsize>(num_groups, heap_size), merge_by_key_kernel(), bulk::root.this_exec, keysSource, valsSource, n, partitionsDevice->get(), num_groups_per_merge, keysDest, valsDest, comp);

    std::swap(keysDest, keysSource);
    std::swap(valsDest, valsSource);
  }

  if(1 & num_passes)
  {
    thrust::copy_n(thrust::cuda::tag(), thrust::make_zip_iterator(thrust::make_tuple(keysSource, valsSource)), n, thrust::make_zip_iterator(thrust::make_tuple(keysDest, valsDest)));
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
  std::cout << "small input: " << std::endl;
  std::cout << "int: " << std::endl;

  validate<int>(20);

  size_t n = 12345678;

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

