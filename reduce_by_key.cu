#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <bulk/bulk.hpp>
#include "head_flags.hpp"
#include "tail_flags.hpp"
#include "time_invocation_cuda.hpp"
#include "reduce_intervals.hpp"


template<typename FlagType, typename ValueType, typename BinaryFunction>
struct scan_head_flags_functor
{
  BinaryFunction binary_op;

  typedef thrust::tuple<FlagType,ValueType> result_type;
  typedef result_type first_argument_type;
  typedef result_type second_argument_type;

  __host__ __device__
  scan_head_flags_functor(BinaryFunction binary_op)
    : binary_op(binary_op)
  {}

  __host__ __device__
  result_type operator()(const first_argument_type &a, const second_argument_type &b)
  {
    ValueType val = thrust::get<0>(b) ? thrust::get<1>(b) : binary_op(thrust::get<1>(a), thrust::get<1>(b));
    FlagType flag = thrust::get<0>(a) + thrust::get<0>(b);
    return result_type(flag, val);
  }
};


template<std::size_t groupsize,
         std::size_t grainsize,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryFunction>
__device__
void scan_head_flags_and_values(bulk::static_execution_group<groupsize,grainsize> &exec,
                                InputIterator1 head_flags_first,
                                InputIterator1 head_flags_last,
                                InputIterator2 values_first,
                                OutputIterator1 head_flags_result,
                                OutputIterator2 values_result,
                                BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<InputIterator1>::type flag_type;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type;

  scan_head_flags_functor<flag_type, value_type, BinaryFunction> f(binary_op);

  bulk::inclusive_scan(exec,
                       thrust::make_zip_iterator(thrust::make_tuple(head_flags_first,  values_first)),
                       thrust::make_zip_iterator(thrust::make_tuple(head_flags_last,   values_first)),
                       thrust::make_zip_iterator(thrust::make_tuple(head_flags_result, values_result)),
                       f);
}


template<std::size_t groupsize,
         std::size_t grainsize,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__device__
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(bulk::static_execution_group<groupsize,grainsize> &exec,
              InputIterator1 keys_first, InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_result,
              OutputIterator2 values_result,
              BinaryPredicate pred,
              BinaryFunction binary_op)
{
  typename thrust::iterator_difference<InputIterator1>::type n = keys_last - keys_first;

  typedef typename thrust::iterator_value<InputIterator1>::type key_type;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type;

  int *scanned_flags = reinterpret_cast<int*>(bulk::malloc(exec, n * sizeof(int)));

  value_type *scanned_values = reinterpret_cast<value_type*>(bulk::malloc(exec, n * sizeof(value_type)));

  head_flags<
    InputIterator1,
    thrust::equal_to<key_type>,
    int
  > flags(keys_first, keys_last);

  scan_head_flags_and_values(exec, flags.begin(), flags.end(), values_first, scanned_flags, scanned_values, binary_op);

  // for each tail element in scanned_flags, the corresponding elements of scanned_values scatters to that flag element - 1
  bulk::scatter_if(exec,
                   thrust::make_zip_iterator(thrust::make_tuple(thrust::reinterpret_tag<thrust::cpp::tag>(keys_first), scanned_values)),
                   thrust::make_zip_iterator(thrust::make_tuple(thrust::reinterpret_tag<thrust::cpp::tag>(keys_last), scanned_values)),
                   thrust::make_transform_iterator(scanned_flags, thrust::placeholders::_1 - 1),
                   make_tail_flags(scanned_flags, scanned_flags + n).begin(),
                   thrust::make_zip_iterator(thrust::make_tuple(keys_result, values_result)));

  int result_size = scanned_flags[n-1];

  bulk::free(exec, scanned_flags);
  bulk::free(exec, scanned_values);

  return thrust::make_pair(keys_result + result_size, values_result + result_size);
}


template<std::size_t groupsize,
         std::size_t grainsize,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename T1,
         typename T2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::tuple<
  OutputIterator1,
  OutputIterator2,
  typename thrust::iterator_value<InputIterator1>::type,
  typename thrust::iterator_value<OutputIterator2>::type
>
__device__
reduce_by_key(bulk::static_execution_group<groupsize,grainsize> &g,
              InputIterator1 keys_first, InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_result,
              OutputIterator2 values_result,
              T1 init_key,
              T2 init_value,
              BinaryPredicate pred,
              BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<InputIterator1>::type key_type;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type; // XXX this should be the type returned by BinaryFunction

  typedef int size_type;

  const size_type interval_size = groupsize * grainsize;

  size_type *s_flags = reinterpret_cast<size_type*>(bulk::malloc(g, interval_size * sizeof(int)));
  value_type *s_values = reinterpret_cast<value_type*>(bulk::malloc(g, interval_size * sizeof(value_type)));

  for(; keys_first < keys_last; keys_first += interval_size, values_first += interval_size)
  {
    // upper bound on n is interval_size
    size_type n = thrust::min<size_type>(interval_size, keys_last - keys_first);

    head_flags_with_init<
      InputIterator1,
      thrust::equal_to<key_type>,
      size_type
    > flags(keys_first, keys_first + n, init_key);

    scan_head_flags_functor<size_type, value_type, BinaryFunction> f(binary_op);

    // load input into smem
    bulk::copy_n(bulk::bound<interval_size>(g),
                 thrust::make_zip_iterator(thrust::make_tuple(flags.begin(), values_first)),
                 n,
                 thrust::make_zip_iterator(thrust::make_tuple(s_flags, s_values)));

    // scan in smem
    bulk::inclusive_scan(bulk::bound<interval_size>(g),
                         thrust::make_zip_iterator(thrust::make_tuple(s_flags,     s_values)),
                         thrust::make_zip_iterator(thrust::make_tuple(s_flags + n, s_values)),
                         thrust::make_zip_iterator(thrust::make_tuple(s_flags,     s_values)),
                         thrust::make_tuple(1, init_value),
                         f);

    // for each tail element in scanned_values, except the last, which is the carry,
    // scatter to that element's corresponding flag element - 1
    // simultaneously scatter the corresponding key
    // XXX can we do this scatter in-place in smem?
    bulk::scatter_if(bulk::bound<interval_size>(g),
                     thrust::make_zip_iterator(thrust::make_tuple(s_values,         thrust::reinterpret_tag<thrust::cpp::tag>(keys_first))),
                     thrust::make_zip_iterator(thrust::make_tuple(s_values + n - 1, thrust::reinterpret_tag<thrust::cpp::tag>(keys_first))),
                     thrust::make_transform_iterator(s_flags, thrust::placeholders::_1 - 1),
                     make_tail_flags(s_flags, s_flags + n).begin(),
                     thrust::make_zip_iterator(thrust::make_tuple(values_result, keys_result)));

    // if the init was not a carry, we need to insert it at the beginning of the result
    if(g.this_exec.index() == 0 && s_flags[0] > 1)
    {
      keys_result[0]   = init_key;
      values_result[0] = init_value;
    }

    size_type result_size = s_flags[n - 1] - 1;

    keys_result    += result_size;
    values_result  += result_size;
    init_key        = keys_first[n-1];
    init_value      = s_values[n - 1];

    g.wait();
  } // end for

  bulk::free(g, s_flags);
  bulk::free(g, s_values);

  return thrust::make_tuple(keys_result, values_result, init_key, init_value);
}


struct reduce_by_key_kernel
{
  template<std::size_t groupsize,
           std::size_t grainsize,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename BinaryPredicate,
           typename BinaryFunction,
           typename Iterator>
  __device__
  void operator()(bulk::static_execution_group<groupsize,grainsize> &g,
                  RandomAccessIterator1 keys_first,
                  RandomAccessIterator1 keys_last,
                  RandomAccessIterator2 values_first,
                  RandomAccessIterator3 keys_result,
                  RandomAccessIterator4 values_result,
                  BinaryPredicate       pred,
                  BinaryFunction        binary_op,
                  Iterator result_size)
  {
    *result_size = ::reduce_by_key(g, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op).first - keys_result;
  }
};


struct reduce_by_key_with_carry_kernel
{
  template<std::size_t groupsize,
           std::size_t grainsize,
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
  void operator()(bulk::static_execution_group<groupsize,grainsize> &g,
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
      reduce_by_key(g,
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
      } // end else

      is_carry[g.index()] = interval_has_carry;
    } // end if
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
  typedef int size_type;

  const difference_type n = keys_last - keys_first;
  const size_type threshold_of_parallelism = 20000;

  if(n <= threshold_of_parallelism)
  {
    thrust::device_vector<size_type> result_size_storage(1);

    // good for 32b types
    bulk::static_execution_group<512,3> g;
    typedef bulk::detail::scan_detail::scan_buffer<512,3,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> heap_type;
    size_type heap_size = sizeof(heap_type);
    bulk::async(bulk::par(g,1,heap_size), reduce_by_key_kernel(), bulk::there, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op, result_size_storage.begin());

    size_type result_size = result_size_storage.front();

    return thrust::make_pair(keys_result + result_size, values_result + result_size);
  } // end if

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type  key_type;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type  value_type;

  // XXX this should be the result of BinaryFunction
  typedef typename thrust::iterator_value<RandomAccessIterator4>::type intermediate_type;

  bulk::static_execution_group<128,5> g;
  size_type tile_size = g.size() * g.grainsize();

  const size_type interval_size = threshold_of_parallelism; 

  size_type subscription = 100;
  size_type num_groups = thrust::min<size_type>(subscription * g.hardware_concurrency(), (n + interval_size - 1) / interval_size);
  uniform_decomposition<size_type> decomp(n, num_groups);

  // count the number of tail flags in each interval
  tail_flags<
    RandomAccessIterator1,
    thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator1>::type>,
    size_type
  > tail_flags(keys_first, keys_last, pred);

  thrust::device_vector<size_type> interval_output_offsets(decomp.size());

  reduce_intervals(tail_flags.begin(), decomp, interval_output_offsets.begin(), thrust::plus<size_type>());

  // scan the interval counts
  thrust::inclusive_scan(interval_output_offsets.begin(), interval_output_offsets.end(), interval_output_offsets.begin());

  // reduce each interval
  thrust::device_vector<bool>              is_carry(decomp.size());
  thrust::device_vector<intermediate_type> interval_values(decomp.size());

  size_type heap_size = tile_size * (sizeof(size_type) + sizeof(value_type));
  bulk::async(bulk::par(g,decomp.size(),heap_size), reduce_by_key_with_carry_kernel(),
    bulk::there, keys_first, decomp, values_first, keys_result, values_result, interval_output_offsets.begin(), interval_values.begin(), is_carry.begin(), thrust::make_tuple(pred, binary_op)
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

  difference_type result_size = interval_output_offsets.back();

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

  std::cerr << "CUDA error: " << cudaGetErrorString(cudaThreadSynchronize()) << std::endl;

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
  size_t n = 12345678;

  validate<int>(n);

  std::cout << "Large input: " << std::endl;
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

