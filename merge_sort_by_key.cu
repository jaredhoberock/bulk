#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/detail/minmax.h>
#include <thrust/random.h>
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
	   typename RandomAccessIterator5,
           typename Compare>
  __device__ void operator()(bulk::concurrent_group<bulk::agent<grainsize>, groupsize> &g, RandomAccessIterator1 keys_first, RandomAccessIterator2 values_first, unsigned int n, RandomAccessIterator3 merge_paths, int num_groups_per_merge, RandomAccessIterator4 keys_result, RandomAccessIterator5 values_result, Compare comp)
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


template<typename Iterator, typename Size, typename Compare>
struct locate_merge_path
{
  Iterator haystack_first;
  Size haystack_size;
  Size num_elements_per_group;
  Size num_groups_per_merge;
  thrust::detail::wrapped_function<Compare,bool> comp;

  locate_merge_path(Iterator haystack_first, Size haystack_size, Size num_elements_per_group, Size num_groups_per_merge, Compare comp)
    : haystack_first(haystack_first),
      haystack_size(haystack_size),
      num_elements_per_group(num_elements_per_group),
      num_groups_per_merge(num_groups_per_merge),
      comp(comp)
  {}

  template<typename Index>
  __host__ __device__
  Index operator()(Index merge_path_idx)
  {
    // find the index of the first group that will participate in the eventual merge
    Size first_group_in_partition = ~(num_groups_per_merge - 1) & merge_path_idx;

    // the size of each group's input
    Size size = num_elements_per_group * (num_groups_per_merge / 2);

    // find pointers to the two input arrays
    Size start1 = num_elements_per_group * first_group_in_partition;
    Size start2 = thrust::min<Size>(haystack_size, start1 + size);

    // the size of each input array
    // note we clamp to the end of the total input to handle the last partial list
    Size n1 = thrust::min<Size>(size, haystack_size - start1);
    Size n2 = thrust::min<Size>(size, haystack_size - start2);
    
    // note that diag is computed as an offset from the beginning of the first list
    Size diag = thrust::min<Size>(n1 + n2, num_elements_per_group * merge_path_idx - start1);

    return bulk::merge_path(haystack_first + start1, n1, haystack_first + start2, n2, diag, comp);
  }
};


template<typename DerivedPolicy, typename Iterator1, typename Size1, typename Iterator2, typename Size2, typename Size3, typename Compare>
void locate_merge_paths_(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                         Iterator1 result,
                         Size1 n,
                         Iterator2 haystack_first,
                         Size2 haystack_size,
                         Size3 num_elements_per_group,
                         Size3 num_groups_per_merge,
                         Compare comp)
{
  locate_merge_path<Iterator2,Size2,Compare> f(haystack_first, haystack_size, num_elements_per_group, num_groups_per_merge, comp);

  thrust::tabulate(exec, result, result + n, f);
}


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void stable_merge_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, Compare comp)
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = keys_last - keys_first;

  if(n <= 0) return;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type;

  typedef int size_type;

  // 78/77/92
  const size_type groupsize = 128;
  const size_type grainsize = 7;
  
  const size_type tilesize = groupsize * grainsize;
  size_type num_groups = (n + tilesize - 1) / tilesize;
  size_type num_passes = thrust::detail::log2_ri(num_groups);

  size_type heap_size = tilesize * thrust::max(sizeof(key_type), sizeof(value_type));
  bulk::async(bulk::grid<groupsize,grainsize>(num_groups, heap_size), stable_sort_each_kernel(), bulk::root.this_exec, keys_first, values_first, n, comp);

  // XXX forward exec from parameters here
  thrust::cuda::tag exec;

  // ping being true means the latest data is in the source array
  bool ping = true;
  thrust::detail::temporary_array<key_type,thrust::cuda::tag>   keys_pong(exec, n);
  thrust::detail::temporary_array<value_type,thrust::cuda::tag> values_pong(exec, n);

  thrust::detail::temporary_array<size_type,thrust::cuda::tag> merge_paths(exec, num_groups + 1);
  
  // merge_by_key_kernel's heap requirements differ
  heap_size = tilesize * thrust::max(sizeof(key_type), sizeof(size_type));

  for(size_type pass = 0; pass < num_passes; ++pass, ping = !ping) 
  {
    size_type num_groups_per_merge = 2 << pass;

    if(ping)
    {
      locate_merge_paths_(exec, merge_paths.begin(), merge_paths.size(), keys_first, n, tilesize, num_groups_per_merge, comp);
      
      bulk::async(bulk::grid<groupsize,grainsize>(num_groups, heap_size), merge_by_key_kernel(), bulk::root.this_exec, keys_first, values_first, n, merge_paths.begin(), num_groups_per_merge, keys_pong.begin(), values_pong.begin(), comp);
    }
    else
    {
      locate_merge_paths_(exec, merge_paths.begin(), merge_paths.size(), keys_pong.begin(), n, tilesize, num_groups_per_merge, comp);
      
      bulk::async(bulk::grid<groupsize,grainsize>(num_groups, heap_size), merge_by_key_kernel(), bulk::root.this_exec, keys_pong.begin(), values_pong.begin(), n, merge_paths.begin(), num_groups_per_merge, keys_first, values_first, comp);
    }
  }

  if(!ping)
  {
    thrust::copy_n(exec, keys_pong.begin(), n,   keys_first);
    thrust::copy_n(exec, values_pong.begin(), n, values_first);
  }
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
  stable_merge_sort_by_key(sorted_keys->begin(), sorted_keys->end(), sorted_values->begin(), my_less());
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

  stable_merge_sort_by_key(sorted_keys.begin(), sorted_keys.end(), sorted_values.begin(), my_less());

  cudaError_t error = cudaThreadSynchronize();
  if(error)
  {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }

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
  for(size_t n = 1; n <= 1 << 20; n <<= 1)
  {
    std::cout << "Testing n = " << n << std::endl;
    validate<int>(n);
  }

  thrust::default_random_engine rng;
  for(int i = 0; i < 20; ++i)
  {
    size_t n = rng() % (1 << 20);
   
    std::cout << "Testing n = " << n << std::endl;
    validate<int>(n);
  }

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

