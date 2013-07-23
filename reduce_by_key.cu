#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/random.h>
#include <bulk/bulk.hpp>
#include "head_flags.hpp"
#include "tail_flags.hpp"
#include "time_invocation_cuda.hpp"
#include "reduce_intervals.hpp"


struct reduce_by_key_kernel
{
  template<typename ConcurrentGroup,
           typename RandomAccessIterator1,
           typename Decomposition,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename RandomAccessIterator5,
           typename RandomAccessIterator6,
           typename RandomAccessIterator7,
           typename BinaryPredicate,
           typename BinaryFunction>
  __device__
  thrust::pair<RandomAccessIterator3,RandomAccessIterator4>
  operator()(ConcurrentGroup &g,
             RandomAccessIterator1 keys_first,
             Decomposition decomp,
             RandomAccessIterator2 values_first,
             RandomAccessIterator3 keys_result,
             RandomAccessIterator4 values_result,
             RandomAccessIterator5 interval_output_offsets,
             RandomAccessIterator6 interval_values,
             RandomAccessIterator7 is_carry,
             //BinaryPredicate pred,
             //BinaryFunction binary_op)
             thrust::tuple<BinaryPredicate,BinaryFunction> pred_and_binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type;
    typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type;

    BinaryPredicate pred = thrust::get<0>(pred_and_binary_op);
    BinaryFunction binary_op = thrust::get<1>(pred_and_binary_op);

    tail_flags<RandomAccessIterator1> tail_flags(keys_first, keys_first + decomp.n(), pred);

    typename Decomposition::size_type input_first, input_last;
    thrust::tie(input_first,input_last) = decomp[g.index()];

    typename Decomposition::size_type output_first = g.index() == 0 ? 0 : interval_output_offsets[g.index() - 1];

    key_type init_key     = keys_first[input_first];
    value_type init_value = values_first[input_first];

    // the inits become the carries
    thrust::tie(keys_result, values_result, init_key, init_value) =
      bulk::reduce_by_key(g,
                          keys_first + input_first + 1,
                          keys_first + input_last,
                          values_first + input_first + 1,
                          keys_result + output_first,
                          values_result + output_first,
                          init_key,
                          init_value,
                          pred,
                          binary_op);

    if(g.this_exec.index() == 0)
    {
      bool interval_has_carry = !tail_flags[input_last-1];

      if(interval_has_carry)
      {
        interval_values[g.index()] = init_value;
      } // end if
      else
      {
        *keys_result   = init_key;
        *values_result = init_value;

        ++keys_result;
        ++values_result;
      } // end else

      is_carry[g.index()] = interval_has_carry;
    } // end if

    return thrust::make_pair(keys_result, values_result);
  }


  template<typename ConcurrentGroup,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename BinaryPredicate,
           typename BinaryFunction,
           typename Iterator>
  __device__
  void operator()(ConcurrentGroup      &g,
                  RandomAccessIterator1 keys_first,
                  RandomAccessIterator1 keys_last,
                  RandomAccessIterator2 values_first,
                  RandomAccessIterator3 keys_result,
                  RandomAccessIterator4 values_result,
                  BinaryPredicate       pred,
                  BinaryFunction        binary_op,
                  Iterator result_size)
  {
    RandomAccessIterator3 old_keys_result = keys_result;

    thrust::tie(keys_result, values_result) =
      operator()(g, keys_first, make_trivial_decomposition(keys_last - keys_first), values_first, keys_result, values_result,
                 thrust::make_constant_iterator<int>(0),
                 thrust::make_discard_iterator(),
                 thrust::make_discard_iterator(),
                 thrust::make_tuple(pred,binary_op));

    if(g.this_exec.index() == 0)
    {
      *result_size = keys_result - old_keys_result;
    }
  }
};


struct tuple_and
{
  typedef bool result_type;

  template<typename Tuple>
  __host__ __device__
  bool operator()(Tuple t)
  {
    return thrust::get<0>(t) && thrust::get<1>(t);
  }
};


template<typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Iterator4,
         typename BinaryFunction>
void sum_tail_carries(Iterator1 interval_values_first,
                      Iterator1 interval_values_last,
                      Iterator2 interval_output_offsets_first,
                      Iterator2 interval_output_offsets_last,
                      Iterator3 is_carry,
                      Iterator4 values_result,
                      BinaryFunction binary_op)
{
  typedef thrust::zip_iterator<thrust::tuple<Iterator2,Iterator3> > zip_iterator;

  tail_flags<zip_iterator> tail_flags(thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets_first, is_carry)),
                                      thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets_last,  is_carry)));

  // for each value in the array of interval values
  //   if it is a carry and it is the tail value in its segment
  //     scatter it to its location in the output array, but sum it together with the value there previously
  thrust::transform_if(interval_values_first, interval_values_last,
                       thrust::make_permutation_iterator(values_result, interval_output_offsets_first),
                       thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(tail_flags.begin(), is_carry)), tuple_and()),
                       thrust::make_permutation_iterator(values_result, interval_output_offsets_first),
                       binary_op,
                       thrust::identity<bool>());
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::pair<RandomAccessIterator3,RandomAccessIterator4>
  my_reduce_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   RandomAccessIterator3 keys_result,
                   RandomAccessIterator4 values_result,
                   BinaryPredicate pred,
                   BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference_type;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type      value_type;
  typedef int size_type;

  const difference_type n = keys_last - keys_first;
  const size_type threshold_of_parallelism = 20000;

  if(n <= threshold_of_parallelism)
  {
    thrust::cuda::tag t;
    thrust::detail::temporary_array<size_type,thrust::cuda::tag> result_size_storage(t, 1);

    // good for 32b types
    const int groupsize = 512;
    const int grainsize = 3;
    size_type heap_size = groupsize * grainsize * (sizeof(size_type) + sizeof(value_type));
    bulk::async(bulk::grid<groupsize,grainsize>(1,heap_size), reduce_by_key_kernel(), bulk::root.this_exec, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op, result_size_storage.begin());

    size_type result_size = result_size_storage[0];

    return thrust::make_pair(keys_result + result_size, values_result + result_size);
  } // end if

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type  key_type;

  // XXX this should be the result of BinaryFunction
  typedef typename thrust::iterator_value<RandomAccessIterator4>::type intermediate_type;

  const size_type groupsize = 128;
  const size_type grainsize = 5;
  size_type tile_size = groupsize * grainsize;

  const size_type interval_size = threshold_of_parallelism; 

  size_type subscription = 100;
  size_type num_groups = thrust::min<size_type>(subscription * bulk::concurrent_group<>::hardware_concurrency(), (n + interval_size - 1) / interval_size);
  uniform_decomposition<size_type> decomp(n, num_groups);

  // count the number of tail flags in each interval
  tail_flags<
    RandomAccessIterator1,
    thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator1>::type>,
    size_type
  > tail_flags(keys_first, keys_last, pred);

  thrust::cuda::tag t;
  thrust::detail::temporary_array<size_type,thrust::cuda::tag> interval_output_offsets(t, decomp.size());

  reduce_intervals(tail_flags.begin(), decomp, interval_output_offsets.begin(), thrust::plus<size_type>());

  // scan the interval counts
  thrust::inclusive_scan(interval_output_offsets.begin(), interval_output_offsets.end(), interval_output_offsets.begin());

  // reduce each interval
  thrust::detail::temporary_array<bool,thrust::cuda::tag> is_carry(t, decomp.size());
  thrust::detail::temporary_array<intermediate_type,thrust::cuda::tag> interval_values(t, decomp.size());

  size_type heap_size = tile_size * (sizeof(size_type) + sizeof(value_type));
  bulk::async(bulk::grid<groupsize,grainsize>(decomp.size(),heap_size), reduce_by_key_kernel(),
    bulk::root.this_exec, keys_first, decomp, values_first, keys_result, values_result, interval_output_offsets.begin(), interval_values.begin(), is_carry.begin(), thrust::make_tuple(pred, binary_op)
  );

  // scan by key the carries
  thrust::inclusive_scan_by_key(thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets.begin(), is_carry.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets.end(),   is_carry.end())),
                                interval_values.begin(),
                                interval_values.begin(),
                                thrust::equal_to<thrust::tuple<size_type,bool> >(),
                                binary_op);

  // sum each tail carry value into the result 
  sum_tail_carries(interval_values.begin(), interval_values.end(),
                   interval_output_offsets.begin(), interval_output_offsets.end(),
                   is_carry.begin(),
                   values_result,
                   binary_op);

  difference_type result_size = interval_output_offsets[interval_output_offsets.size() - 1];

  return thrust::make_pair(keys_result + result_size, values_result + result_size);
}


template<typename T>
size_t my_reduce_by_key(const thrust::device_vector<T> *keys,
                        const thrust::device_vector<T> *values,
                        thrust::device_vector<T> *keys_result,
                        thrust::device_vector<T> *values_result)
{
  return my_reduce_by_key(keys->begin(), keys->end(),
                          values->begin(),
                          keys_result->begin(),
                          values_result->begin(),
                          thrust::equal_to<T>(),
                          thrust::plus<T>()).first -
         keys_result->begin();
}


template<typename T>
size_t thrust_reduce_by_key(const thrust::device_vector<T> *keys,
                            const thrust::device_vector<T> *values,
                            thrust::device_vector<T> *keys_result,
                            thrust::device_vector<T> *values_result)
{
  return thrust::reduce_by_key(keys->begin(), keys->end(),
                               values->begin(),
                               keys_result->begin(),
                               values_result->begin()).first -
         keys_result->begin();
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

    return x % 10;
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
  thrust::device_vector<T> keys(n), values(n);
  thrust::device_vector<T> keys_result(n), values_result(n);

  random_fill(keys);
  random_fill(values);

  size_t my_size = my_reduce_by_key(&keys, &values, &keys_result, &values_result);
  double my_msecs = time_invocation_cuda(50, my_reduce_by_key<T>, &keys, &values, &keys_result, &values_result);

  thrust_reduce_by_key(&keys, &values, &keys_result, &values_result);
  double thrust_msecs = time_invocation_cuda(50, thrust_reduce_by_key<T>, &keys, &values, &keys_result, &values_result);

  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;
  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;

  double my_secs = my_msecs / 1000;

  std::cout << double(n) / my_secs << " keys per second" << std::endl;

  double in_bytes = 2 *sizeof(T) * n;
  double out_bytes = 2 * sizeof(T) * my_size;

  double gigabytes = (in_bytes + out_bytes) / (1 << 30);

  std::cout << gigabytes / my_secs << "GB/s" << std::endl;
  std::cout << "Output ratio: " << double(my_size) / double(n) << std::endl;
}


template<typename T>
void validate(size_t n)
{
  thrust::device_vector<T> keys(n), values(n);
  thrust::device_vector<T> keys_result(n), values_result(n);

  random_fill(keys);
  random_fill(values);

  thrust::device_vector<T> keys_ref(n), values_ref(n);
  size_t thrust_size = thrust_reduce_by_key(&keys, &values, &keys_ref, &values_ref);
  keys_ref.resize(thrust_size);
  values_ref.resize(thrust_size);

  size_t my_size = my_reduce_by_key(&keys, &values, &keys_result, &values_result);
  keys_result.resize(my_size);
  values_result.resize(my_size);

  cudaError_t error = cudaDeviceSynchronize();
  if(error)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }

  if(values_result != values_ref && n < 30)
  {
    std::cout << "values_result: ";
    thrust::copy(values_result.begin(), values_result.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl << std::endl;


    std::cout << "values_ref:    ";
    thrust::copy(values_ref.begin(), values_ref.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl << std::endl;
  }

  assert(keys_result == keys_ref);
  assert(values_result == values_ref);
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
  std::cout << std::endl;

  std::cout << "float: " << std::endl;
  compare<float>(n);
  std::cout << std::endl;

  std::cout << "double: " << std::endl;
  compare<double>(n);
  std::cout << std::endl << std::endl;

  n = 19999;
  std::cout << "Small input: " << std::endl;
  std::cout << "int: " << std::endl;
  compare<int>(n);
  std::cout << std::endl;

  std::cout << "float: " << std::endl;
  compare<float>(n);
  std::cout << std::endl;

  std::cout << "double: " << std::endl;
  compare<double>(n);
  std::cout << std::endl;

  return 0;
}

