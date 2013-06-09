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
#include <thrust/copy.h>
#include <bulk/bulk.hpp>


template<unsigned int size, unsigned int grainsize>
struct inclusive_scan_n
{
  template<typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<size,grainsize> &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk::inclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};


template<unsigned int size, unsigned int grainsize>
struct exclusive_scan_n
{
  template<typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<size,grainsize> &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
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


template<std::size_t groupsize, std::size_t grainsize>
struct inclusive_downsweep
{
  template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<groupsize,grainsize> &this_group,
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


template<bool commutative, std::size_t groupsize, std::size_t grainsize>
struct reduce_tiles
{
  template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename BinaryFunction>
  __device__ void operator()(bulk::static_thread_group<groupsize,grainsize> &this_group,
                             RandomAccessIterator1 first,
                             Size n,
                             Size partition_size,
                             Size last_partial_partition_size,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
    
    thrust::pair<Size,Size> range = tile_range<Size>(this_group.index(), partition_size, last_partial_partition_size, groupsize * grainsize, n);

    // it's much faster to pass the last value as the init for some reason
    value_type init = first[range.second-1];

    value_type total = commutative ?
      bulk::reduce(this_group, first + range.first, first + range.second - 1, init, binary_op) :
      bulk::noncommutative_reduce(this_group, first + range.first, first + range.second - 1, init, binary_op);

    if(this_group.this_thread.index() == 0)
    {
      result[this_group.index()] = total;
    } // end if
  } // end operator()
}; // end reduce_tiles


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
RandomAccessIterator2 inclusive_scan(RandomAccessIterator1 first, RandomAccessIterator1 last, RandomAccessIterator2 result, T init, BinaryFunction binary_op)
{
  // XXX TODO pass explicit heap sizes

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
    const int groupsize = 512;
    const int grainsize = 3;

    bulk::static_thread_group<groupsize,grainsize> group;
    bulk::async(bulk::par(group, 1), inclusive_scan_n<groupsize,grainsize>(), bulk::there, first, n, result, init, binary_op);
  } // end if
  else
  {
    // Run the parallel raking reduce as an upsweep.
    const int groupsize1 = 128;
    const int grainsize1 = 7;
    bulk::static_thread_group<groupsize1,grainsize1> group1;

    const Size partition_size = groupsize1 * grainsize1;
    
    Size num_partitions = (n + partition_size - 1) / partition_size;
    Size num_groups = thrust::min<Size>(group1.hardware_concurrency() * 25, num_partitions);

    // each group consumes one tile of data
    Size num_partitions_per_tile = num_partitions / num_groups;
    Size last_partial_partition_size = num_partitions % num_groups;
    
    thrust::cuda::tag t;
    thrust::detail::temporary_array<intermediate_type,thrust::cuda::tag> carries(t, num_groups);
    	
    // n loads + num_groups stores
    const bool commutative = thrust::detail::is_commutative<BinaryFunction>::value;
    bulk::async(bulk::par(group1,num_groups), reduce_tiles<commutative,groupsize1,grainsize1>(), bulk::there, first, n, num_partitions_per_tile, last_partial_partition_size, carries.begin(), binary_op);
    
    // scan the sums to get the carries
    const int groupsize2 = 256;
    const int grainsize2 = 3;

    // num_groups loads + num_groups stores
    bulk::static_thread_group<groupsize2,grainsize2> group2;
    bulk::async(bulk::par(group2,1), exclusive_scan_n<groupsize2,grainsize2>(), bulk::there, carries.begin(), num_groups, carries.begin(), init, binary_op);
    
    // do the downsweep - n loads, n stores
    bulk::async(bulk::par(group1,num_groups), inclusive_downsweep<groupsize1,grainsize1>(), bulk::there, first, n, num_partitions_per_tile, last_partial_partition_size, carries.begin(), result, binary_op);
  } // end else

  return result + n;
} // end inclusive_scan()


typedef int T;


void my_scan(thrust::device_vector<T> *data, T init)
{
  ::inclusive_scan(data->begin(), data->end(), data->begin(), init, thrust::plus<int>());
}


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

  ::inclusive_scan(d_input.begin(), d_input.end(), d_result.begin(), init, thrust::plus<int>());

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

  thrust::device_vector<T> vec(1 << 28);

  sean_scan(&vec);
  double sean_msecs = time_invocation_cuda(50, sean_scan, &vec);

  my_scan(&vec, 13);
  double my_msecs = time_invocation_cuda(50, my_scan, &vec, 13);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "My time: " << my_msecs << " ms" << std::endl;

  std::cout << "My relative performance: " << sean_msecs / my_msecs << std::endl;

  return 0;
}

