#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/detail/range/tail_flags.h>
#include <thrust/iterator/reverse_iterator.h>
#include <bulk/bulk.hpp>
#include "tail_flags.hpp"
#include "time_invocation_cuda.hpp"


struct reduce_intervals
{
  template<typename ExecutionGroup, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename BinaryFunction>
  __device__ void operator()(ExecutionGroup &this_group,
                             RandomAccessIterator1 first,
                             Size n,
                             Size interval_size,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type;

    Size start = this_group.index() * interval_size;
    Size end   = thrust::min<Size>(n, start + interval_size);

    // the last group has zero input and returns a 0 sum
    value_type init = 0;
    value_type sum = bulk::reduce(this_group, first + start, first + end, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      result[this_group.index()] = sum;
    } // end if
  } // end operator()
}; // end reduce_intervals


template<typename InputIterator, typename BinaryFunction, typename OutputIterator = void>
  struct partial_sum_type
    : thrust::detail::eval_if<
        thrust::detail::has_result_type<BinaryFunction>::value,
        thrust::detail::result_type<BinaryFunction>,
        thrust::detail::eval_if<
          thrust::detail::is_output_iterator<OutputIterator>::value,
          thrust::iterator_value<InputIterator>,
          thrust::iterator_value<OutputIterator>
        >
      >
{};


template<typename InputIterator, typename BinaryFunction>
  struct partial_sum_type<InputIterator,BinaryFunction,void>
    : thrust::detail::eval_if<
        thrust::detail::has_result_type<BinaryFunction>::value,
        thrust::detail::result_type<BinaryFunction>,
        thrust::iterator_value<InputIterator>
      >
{};


template<typename InputIterator1,
         typename InputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  thrust::tuple<
    InputIterator1,
    typename InputIterator1::value_type,
    typename partial_sum_type<InputIterator2,BinaryFunction>::type
  >
    reduce_last_segment_backward(InputIterator1 keys_first,
                                 InputIterator1 keys_last,
                                 InputIterator2 values_first,
                                 BinaryPredicate binary_pred,
                                 BinaryFunction binary_op)
{
  typename thrust::iterator_difference<InputIterator1>::type n = keys_last - keys_first;

  // reverse the ranges and consume from the end
  thrust::reverse_iterator<InputIterator1> keys_first_r(keys_last);
  thrust::reverse_iterator<InputIterator1> keys_last_r(keys_first);
  thrust::reverse_iterator<InputIterator2> values_first_r(values_first + n);

  typename InputIterator1::value_type result_key = *keys_first_r;
  typename partial_sum_type<InputIterator2,BinaryFunction>::type result_value = *values_first_r;

  // consume the entirety of the first key's sequence
  for(++keys_first_r, ++values_first_r;
      (keys_first_r != keys_last_r) && binary_pred(*keys_first_r, result_key);
      ++keys_first_r, ++values_first_r)
  {
    result_value = binary_op(result_value, *values_first_r);
  }

  return thrust::make_tuple(keys_first_r.base(), result_key, result_value);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  thrust::tuple<
    OutputIterator1,
    OutputIterator2,
    typename InputIterator1::value_type,
    typename partial_sum_type<InputIterator2,BinaryFunction>::type
  >
    reduce_by_key_with_carry(InputIterator1 keys_first, 
                             InputIterator1 keys_last,
                             InputIterator2 values_first,
                             OutputIterator1 keys_output,
                             OutputIterator2 values_output,
                             BinaryPredicate binary_pred,
                             BinaryFunction binary_op)
{
  // first, consume the last sequence to produce the carry
  // XXX is there an elegant way to pose this such that we don't need to default construct the carries?
  typename InputIterator1::value_type carry_key;
  typename partial_sum_type<InputIterator2,BinaryFunction>::type carry_value;

  thrust::tie(keys_last, carry_key, carry_value) = reduce_last_segment_backward(keys_first, keys_last, values_first, binary_pred, binary_op);

  // finish with sequential reduce_by_key
  thrust::cpp::tag seq;
  thrust::tie(keys_output, values_output) =
    thrust::reduce_by_key(seq, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
  
  return thrust::make_tuple(keys_output, values_output, carry_key, carry_value);
}


template<typename Iterator>
__host__ __device__
  bool interval_has_carry(size_t interval_idx, size_t interval_size, size_t num_intervals, Iterator tail_flags)
{
  // to discover whether the interval has a carry, look at the tail_flag corresponding to its last element 
  // the final interval never has a carry by definition
  return (interval_idx + 1 < num_intervals) ? !tail_flags[(interval_idx + 1) * interval_size - 1] : false;
}


template<typename InputIterator1,
         typename Size,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2,
         typename OutputIterator3,
         typename OutputIterator4,
         typename BinaryPredicate,
         typename BinaryFunction>
__global__ void reduce_by_key_kernel(InputIterator1 keys_first,
                                     Size n,
                                     Size num_intervals,
                                     Size interval_size,
                                     InputIterator2 values_first,
                                     InputIterator3 interval_output_offsets,
                                     OutputIterator1 carry_keys,
                                     OutputIterator2 carry_values,
                                     OutputIterator3 keys_result,
                                     OutputIterator4 values_result,
                                     BinaryPredicate pred,
                                     BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<InputIterator1>::type key_type;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type;

  if(threadIdx.x == 0)
  {
    tail_flags<InputIterator1,BinaryPredicate> tail_flags(keys_first, keys_first + n, pred);

    int i = blockIdx.x;

    int input_idx = i * interval_size;
    int input_end = thrust::min(n, input_idx + interval_size);
    int output_idx = interval_output_offsets[i];
    
    // reduce this interval, and grab the final segment's key & sum
    OutputIterator1 final_key_result;
    OutputIterator2 final_value_result;
    key_type carry_key;
    value_type carry_value;
    
    thrust::tie(final_key_result, final_value_result, carry_key, carry_value) =
      reduce_by_key_with_carry(keys_first    + input_idx,
                               keys_first    + input_end,
                               values_first  + input_idx,
                               keys_result   + output_idx,
                               values_result + output_idx,
                               pred,
                               binary_op);
    carry_keys[i] = carry_key;
    carry_values[i] = carry_value;
    
    // XXX another way to do this would be to compare output_idx to interval_output_offsets[i+1]
    //     if they differ, then this interval's end coincides with the end of a segment
    if(!interval_has_carry(i, interval_size, num_intervals, tail_flags.begin()))
    {
      *final_key_result = carry_key;
      *final_value_result = carry_value;
    } // end else
  } // end if
} // end reduce_by_key_kernel()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::pair<OutputIterator1,OutputIterator2>
my_reduce_by_key(InputIterator1 keys_first, 
                 InputIterator1 keys_last,
                 InputIterator2 values_first,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 BinaryPredicate pred,
                 BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  difference_type n = keys_last - keys_first;
  if(n == 0) return thrust::make_pair(keys_result, values_result);

  difference_type interval_size = 10000;
  difference_type num_intervals = (n + (interval_size - 1)) / interval_size;

  // the last element stores the total output size
  thrust::device_vector<difference_type> interval_output_offsets(num_intervals + 1);

  tail_flags<InputIterator1,BinaryPredicate> tail_flags(keys_first, keys_last, pred);

  // count the number of tail flags in each interval
  // XXX needs tuning
  bulk::static_execution_group<256,5> g;
  bulk::async(bulk::par(g, num_intervals + 1), reduce_intervals(), bulk::there, tail_flags.begin(), n, interval_size, interval_output_offsets.begin(), thrust::plus<difference_type>());

  // scan the interval counts to get output offsets
  thrust::exclusive_scan(interval_output_offsets.begin(), interval_output_offsets.end(), interval_output_offsets.begin(), 0, thrust::plus<difference_type>());

  typedef typename thrust::iterator_value<InputIterator1>::type key_type;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type;

  thrust::device_vector<key_type>   carry_keys(num_intervals);
  thrust::device_vector<value_type> carry_values(num_intervals);

  reduce_by_key_kernel<<<num_intervals,4>>>(keys_first, n, num_intervals, interval_size, values_first, interval_output_offsets.begin(), carry_keys.begin(), carry_values.begin(), keys_result, values_result, pred, binary_op);

  for(int i = 0; i < carry_values.size(); ++i)
  {
    if(interval_has_carry(i, interval_size, num_intervals, tail_flags.begin()))
    {
      int output_idx = interval_output_offsets[i+1];

      values_result[output_idx] = binary_op(values_result[output_idx], carry_values[i]);
    } // end if
  } // end for i

  difference_type size_of_result = interval_output_offsets.back();
  return thrust::make_pair(keys_result + size_of_result, values_result + size_of_result);
} // end my_reduce_by_key()


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

  my_reduce_by_key(&keys, &values, &keys_result, &values_result);
  double my_msecs = time_invocation_cuda(50, my_reduce_by_key<T>, &keys, &values, &keys_result, &values_result);

  thrust_reduce_by_key(&keys, &values, &keys_result, &values_result);
  double thrust_msecs = time_invocation_cuda(50, thrust_reduce_by_key<T>, &keys, &values, &keys_result, &values_result);

  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;

  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;
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

  std::cout << "CUDA error: " << cudaGetErrorString(cudaThreadSynchronize()) << std::endl;

  assert(keys_result == keys_ref);
  assert(values_result == values_ref);
}


int main()
{
  //size_t n = 123456789;
  //size_t n = 10001;
  size_t n = 100010;

  validate<int>(n);

//  std::cout << "Large input: " << std::endl;
//  std::cout << "int: " << std::endl;
//  compare<int>(n);
//
//  std::cout << "float: " << std::endl;
//  compare<float>(n);
//
//  std::cout << "double: " << std::endl;
//  compare<double>(n);
//  std::cout << std::endl;

  return 0;
}

