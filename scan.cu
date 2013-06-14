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
#include <thrust/detail/type_traits/function_traits.h>
#include <bulk/bulk.hpp>


struct inclusive_scan_n
{
  template<std::size_t size, std::size_t grainsize, typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(bulk::static_execution_group<size,grainsize> &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk::inclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};

struct exclusive_scan_n
{
  template<std::size_t size, std::size_t grainsize, typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(bulk::static_execution_group<size,grainsize> &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk::exclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};


// compute the indices of the first and last (exclusive) partitions the group will consume
template<typename Size>
__device__
thrust::pair<Size,Size> tile_range_in_partitions(Size tile_index,
                                                 Size num_partitions_per_tile,
                                                 Size last_partial_partition_size)
{
  thrust::pair<Size,Size> result;

  result.first = num_partitions_per_tile * tile_index;
  result.first += thrust::min<Size>(tile_index, last_partial_partition_size);

  result.second = result.first + num_partitions_per_tile + (tile_index < last_partial_partition_size);

  return result;
} // end tile_range_in_partitions()


// compute the indices of the first and last (exclusive) elements the group will consume
template<typename Size>
__device__
thrust::pair<Size,Size> tile_range(Size tile_index,
                                   Size num_partitions_per_group,
                                   Size last_partial_partition_size,
                                   Size partition_size,
                                   Size n)
{
  thrust::pair<Size,Size> result = tile_range_in_partitions(tile_index, num_partitions_per_group, last_partial_partition_size);
  result.first *= partition_size;
  result.second = thrust::min<Size>(n, result.second * partition_size);
  return result;
} // end tile_range()


struct inclusive_downsweep
{
  template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename BinaryFunction>
  __device__ void operator()(bulk::static_execution_group<groupsize,grainsize> &this_group,
                             RandomAccessIterator1 first,
                             Size n,
                             Size num_partitions_per_tile,
                             Size last_partial_partition_size,
                             RandomAccessIterator2 carries_first,
                             RandomAccessIterator3 result,
                             BinaryFunction binary_op)
  {
    const Size partition_size = groupsize * grainsize;
  
    thrust::pair<Size,Size> range = tile_range<Size>(this_group.index(), num_partitions_per_tile, last_partial_partition_size, partition_size, n);
  
    RandomAccessIterator1 last = first + range.second;
    first += range.first;
    result += range.first;
  
    typename thrust::iterator_value<RandomAccessIterator2>::type carry = carries_first[this_group.index()];

    bulk::inclusive_scan(this_group, first, last, result, carry, binary_op);
  }
};


struct accumulate_tiles
{
  template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename BinaryFunction>
  __device__ void operator()(bulk::static_execution_group<groupsize,grainsize> &this_group,
                             RandomAccessIterator1 first,
                             Size n,
                             Size num_partitions_per_tile,
                             Size last_partial_partition_size,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
    
    thrust::pair<Size,Size> range = tile_range<Size>(this_group.index(), num_partitions_per_tile, last_partial_partition_size, groupsize * grainsize, n);

    const bool commutative = thrust::detail::is_commutative<BinaryFunction>::value;

    // for a commutative accumulate, it's much faster to pass the last value as the init for some reason
    value_type init = commutative ? first[range.second-1] : *first;

    value_type sum = commutative ?
      bulk::accumulate(this_group, first + range.first, first + range.second - 1, init, binary_op) :
      bulk::accumulate(this_group, first + range.first + 1, first + range.second, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      result[this_group.index()] = sum;
    } // end if
  } // end operator()
}; // end accumulate_tiles


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
RandomAccessIterator2 inclusive_scan(RandomAccessIterator1 first, RandomAccessIterator1 last, RandomAccessIterator2 result, T init, BinaryFunction binary_op)
{
  typedef typename bulk::detail::scan_detail::scan_intermediate<
    RandomAccessIterator1,
    RandomAccessIterator2,
    BinaryFunction
  >::type intermediate_type;

  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type Size;

  Size n = last - first;
  
  const Size threshold_of_parallelism = 20000;

  if(n < threshold_of_parallelism)
  {
    bulk::static_execution_group<512,3> group;
    typedef bulk::detail::scan_detail::scan_buffer<512,3,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> heap_type;
    Size heap_size = sizeof(heap_type);
    bulk::async(bulk::par(group, 1, heap_size), inclusive_scan_n(), bulk::there, first, n, result, init, binary_op);
  } // end if
  else
  {
    // Run the parallel raking reduce as an upsweep.
    bulk::static_execution_group<128,7> group1;

    const Size partition_size = group1.size() * group1.grainsize();
    
    Size num_partitions = (n + partition_size - 1) / partition_size;

    // 20 determined from empirical testing on k20c & GTX 480
    int subscription = 20;
    Size num_groups = thrust::min<Size>(subscription * group1.hardware_concurrency(), num_partitions);

    // each group consumes one tile of data
    Size num_partitions_per_tile = num_partitions / num_groups;
    Size last_partial_partition_size = num_partitions % num_groups;
    
    thrust::cuda::tag t;
    thrust::detail::temporary_array<intermediate_type,thrust::cuda::tag> carries(t, num_groups);
    	
    // n loads + num_groups stores
    Size heap_size = group1.size() * sizeof(intermediate_type);
    bulk::async(bulk::par(group1,num_groups,heap_size), accumulate_tiles(), bulk::there, first, n, num_partitions_per_tile, last_partial_partition_size, carries.begin(), binary_op);
    
    // scan the sums to get the carries
    // num_groups loads + num_groups stores
    bulk::static_execution_group<256,3> group2;
    typedef bulk::detail::scan_detail::scan_buffer<256,3,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> heap_type2;
    heap_size = sizeof(heap_type2);
    bulk::async(bulk::par(group2,1,heap_size), exclusive_scan_n(), bulk::there, carries.begin(), num_groups, carries.begin(), init, binary_op);

    // do the downsweep - n loads, n stores
    typedef bulk::detail::scan_detail::scan_buffer<128,7,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> heap_type3;
    heap_size = sizeof(heap_type3);
    bulk::async(bulk::par(group1,num_groups,heap_size), inclusive_downsweep(), bulk::there, first, n, num_partitions_per_tile, last_partial_partition_size, carries.begin(), result, binary_op);
  } // end else

  return result + n;
} // end inclusive_scan()


template<typename T>
void my_scan(thrust::device_vector<T> *data, T init)
{
  ::inclusive_scan(data->begin(), data->end(), data->begin(), init, thrust::plus<T>());
}


template<typename T>
void do_it(size_t n)
{
  thrust::host_vector<T> h_input(n);
  thrust::fill(h_input.begin(), h_input.end(), 1);

  thrust::host_vector<T> h_result(n);

  T init = 13;

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_result.begin());
  thrust::for_each(h_result.begin(), h_result.end(), thrust::placeholders::_1 += init);

  thrust::device_vector<T> d_input = h_input;
  thrust::device_vector<T> d_result(d_input.size());

  ::inclusive_scan(d_input.begin(), d_input.end(), d_result.begin(), init, thrust::plus<T>());

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
  typedef typename thrust::iterator_value<InputIterator>::type T;

  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);

  mgpu::Scan<mgpu::MgpuScanTypeInc>(thrust::raw_pointer_cast(&*first),
                                    last - first,
                                    thrust::raw_pointer_cast(&*result),
                                    mgpu::ScanOp<mgpu::ScanOpTypeAdd,T>(),
                                    (T*)0,
                                    false,
                                    *ctx);

  return result + (last - first);
}


template<typename T>
void sean_scan(thrust::device_vector<T> *data)
{
  mgpu_inclusive_scan(data->begin(), data->end(), data->begin());
}


template<typename T>
void thrust_scan(thrust::device_vector<T> *data)
{
  thrust::inclusive_scan(data->begin(), data->end(), data->begin());
}


template<typename T>
void compare()
{
  thrust::device_vector<T> vec(1 << 28);

  sean_scan(&vec);
  double sean_msecs = time_invocation_cuda(50, sean_scan<T>, &vec);

  thrust_scan(&vec);
  double thrust_msecs = time_invocation_cuda(50, thrust_scan<T>, &vec);

  my_scan(&vec, T(13));
  double my_msecs = time_invocation_cuda(50, my_scan<T>, &vec, 13);

  std::cout << "Sean's time:   " << sean_msecs << " ms" << std::endl;
  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;

  std::cout << "Performance relative to Sean:   " << sean_msecs / my_msecs << std::endl;
  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;
}



int main()
{
  for(size_t n = 1; n <= 1 << 20; n <<= 1)
  {
    std::cout << "Testing n = " << n << std::endl;
    do_it<int>(n);
  }

  thrust::default_random_engine rng;
  for(int i = 0; i < 20; ++i)
  {
    size_t n = rng() % (1 << 20);
   
    std::cout << "Testing n = " << n << std::endl;
    do_it<int>(n);
  }

  std::cout << "32b int:" << std::endl;
  compare<int>();

  std::cout << "64b float:" << std::endl;
  compare<double>();

  return 0;
}

