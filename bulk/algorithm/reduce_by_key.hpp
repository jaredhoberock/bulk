#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_group.hpp>
#include <bulk/algorithm/copy.hpp>
#include <bulk/algorithm/scan.hpp>
#include <bulk/algorithm/scatter.hpp>
#include <bulk/malloc.hpp>
#include <bulk/algorithm/detail/head_flags.hpp>
#include <bulk/algorithm/detail/tail_flags.hpp>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/minmax.h>


BULK_NS_PREFIX
namespace bulk
{
namespace detail
{
namespace reduce_by_key_detail
{
} // end reduce_by_key_detail
} // end detail


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
} // end reduce_by_key()


} // end bulk
BULK_NS_SUFFIX

