#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/detail/range/tail_flags.h>
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

  key_type carry_key;
  value_type carry_value;

  for(int i = 0; i < num_intervals; ++i)
  {
    int input_idx = i * interval_size;
    int input_end = thrust::min(n, input_idx + interval_size);
    int output_idx = interval_output_offsets[i];

    if(i == 0)
    {
      carry_key = keys_first[input_idx];
      carry_value = values_first[input_idx];

      ++input_idx;
    } // end if

    for(; input_idx != input_end; ++input_idx)
    {
      key_type key = keys_first[input_idx];
      value_type value = values_first[input_idx];

      if(pred(carry_key, key))
      {
        carry_value = binary_op(carry_value, value);
      } // end if
      else
      {
        // new segment begins

        // store the result of the previous segment
        keys_result[output_idx] = carry_key;
        values_result[output_idx] = carry_value;

        ++output_idx;

        carry_key = key;
        carry_value = value;
      } // end else
    } // end for

    // store the final segment's sum
    // XXX we have to be careful about whether we do this
    //     we don't want to store this sum unless the next
    //     interval begins a new segment
    keys_result[output_idx] = carry_key;
    values_result[output_idx] = carry_value;
  } // end for

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
  size_t n = 10001;

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

