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
#include "decomposition.hpp"


struct inclusive_scan_n
{
  template<typename ConcurrentGroup, typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk::inclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};

struct exclusive_scan_n
{
  template<typename ConcurrentGroup, typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk::exclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};


struct inclusive_downsweep
{
  template<typename ConcurrentGroup, typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename RandomAccessIterator3, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group,
                             RandomAccessIterator1 first,
                             Decomposition decomp,
                             RandomAccessIterator2 carries_first,
                             RandomAccessIterator3 result,
                             BinaryFunction binary_op)
  {
    typename Decomposition::range range = decomp[this_group.index()];
  
    RandomAccessIterator1 last = first + range.second;
    first += range.first;
    result += range.first;
  
    typename thrust::iterator_value<RandomAccessIterator2>::type carry = carries_first[this_group.index()];

    bulk::inclusive_scan(this_group, first, last, result, carry, binary_op);
  }
};


struct accumulate_tiles
{
  template<typename ConcurrentGroup, typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group,
                             RandomAccessIterator1 first,
                             Decomposition decomp,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
    
    typename Decomposition::range range = decomp[this_group.index()];

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
    typedef bulk::detail::scan_detail::scan_buffer<512,3,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> heap_type;
    Size heap_size = sizeof(heap_type);
    bulk::async(bulk::par(bulk::con<512,3>(heap_size), 1), inclusive_scan_n(), bulk::root.this_exec, first, n, result, init, binary_op);
  } // end if
  else
  {
    // determined from empirical testing on k20c
    const int groupsize = sizeof(intermediate_type) <= sizeof(int) ? 128 : 256;
    const int grainsize = sizeof(intermediate_type) <= sizeof(int) ?   9 :   5;

    const Size tile_size = groupsize * grainsize;
    int num_tiles = (n + tile_size - 1) / tile_size;

    // 20 determined from empirical testing on k20c & GTX 480
    int subscription = 20;
    Size num_groups = thrust::min<Size>(subscription * bulk::concurrent_group<>::hardware_concurrency(), num_tiles);

    aligned_decomposition<Size> decomp(n, num_groups, tile_size);

    thrust::cuda::tag t;
    thrust::detail::temporary_array<intermediate_type,thrust::cuda::tag> carries(t, num_groups);
    	
    // Run the parallel raking reduce as an upsweep.
    // n loads + num_groups stores
    Size heap_size = groupsize * sizeof(intermediate_type);
    bulk::async(bulk::grid<groupsize,grainsize>(num_groups,heap_size), accumulate_tiles(), bulk::root.this_exec, first, decomp, carries.begin(), binary_op);
    
    // scan the sums to get the carries
    // num_groups loads + num_groups stores
    typedef bulk::detail::scan_detail::scan_buffer<256,3,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> heap_type2;
    heap_size = sizeof(heap_type2);
    bulk::async(bulk::con<256,3>(heap_size), exclusive_scan_n(), bulk::root.this_exec, carries.begin(), num_groups, carries.begin(), init, binary_op);

    // do the downsweep - n loads, n stores
    typedef bulk::detail::scan_detail::scan_buffer<
      groupsize,
      grainsize,
      RandomAccessIterator1,RandomAccessIterator2,BinaryFunction
    > heap_type3;
    heap_size = sizeof(heap_type3);
    bulk::async(bulk::grid<groupsize,grainsize>(num_groups,heap_size), inclusive_downsweep(), bulk::root.this_exec, first, decomp, carries.begin(), result, binary_op);
  } // end else

  return result + n;
} // end inclusive_scan()


template<typename T>
void my_scan(thrust::device_vector<T> *data, T init)
{
  ::inclusive_scan(data->begin(), data->end(), data->begin(), init, thrust::plus<T>());
}


template<typename T>
void validate(size_t n)
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


template<typename T>
void thrust_scan(thrust::device_vector<T> *data)
{
  thrust::inclusive_scan(data->begin(), data->end(), data->begin());
}


template<typename T>
void compare(size_t n = 1 << 28)
{
  thrust::device_vector<T> vec(n);

  thrust_scan(&vec);
  double thrust_msecs = time_invocation_cuda(50, thrust_scan<T>, &vec);

  my_scan(&vec, T(13));
  double my_msecs = time_invocation_cuda(50, my_scan<T>, &vec, 13);

  std::cout << "N: " << n << std::endl;
  std::cout << "  Thrust's time:                  " << thrust_msecs << " ms" << std::endl;
  std::cout << "  My time:                        " << my_msecs << " ms" << std::endl;
  std::cout << "  Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;
  std::cout << std::endl;
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

  std::cout << "32b int:" << std::endl;
  for(int i = 0; i < 28; ++i)
  {
    compare<int>(1 << i);
  }

  std::cout << "64b float:" << std::endl;
  for(int i = 0; i < 28; ++i)
  {
    compare<double>(1 << i);
  }

  return 0;
}

