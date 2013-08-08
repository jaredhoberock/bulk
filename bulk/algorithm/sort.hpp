#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_policy.hpp>
#include <thrust/detail/swap.h>

BULK_NS_PREFIX
namespace bulk
{
namespace detail
{
namespace sort_detail
{


template<int i, int bound>
struct stable_odd_even_transpose_sort_by_key_impl
{
  template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
  static __device__
  void sort(RandomAccessIterator1 keys, RandomAccessIterator2 values, int n, Compare comp)
  {
    #pragma unroll
    for(int j = 1 & i; j < bound - 1; j += 2)
    {
      if(j + 1 < n && comp(keys[j + 1], keys[j]))
      {
        using thrust::swap;

      	swap(keys[j], keys[j + 1]);
      	swap(values[j], values[j + 1]);
      }
    }

    stable_odd_even_transpose_sort_by_key_impl<i + 1, bound>::sort(keys, values, n, comp);
  }
};


template<int i> struct stable_odd_even_transpose_sort_by_key_impl<i, i>
{
  template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
  static __device__ void sort(RandomAccessIterator1, RandomAccessIterator2, int, Compare) { }
};


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
__forceinline__ __device__
void stable_odd_even_transpose_sort_by_key(const bounded<bound,agent<grainsize> > &,
                                           RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                                           RandomAccessIterator2 values_first,
                                           Compare comp)
{
  stable_odd_even_transpose_sort_by_key_impl<0, bound>::sort(keys_first, values_first, keys_last - keys_first, comp);
} // end stable_odd_even_transpose_sort_by_key()


template<int i, int bound>
struct stable_odd_even_transpose_sort_impl
{
  template<typename RandomAccessIterator, typename Compare>
  static __device__
  void sort(RandomAccessIterator keys, int n, Compare comp)
  {
    #pragma unroll
    for(int j = 1 & i; j < bound - 1; j += 2)
    {
      if(j + 1 < n && comp(keys[j + 1], keys[j]))
      {
        using thrust::swap;

      	swap(keys[j], keys[j + 1]);
      }
    }

    stable_odd_even_transpose_sort_impl<i + 1, bound>::sort(keys, n, comp);
  }
};


template<int i> struct stable_odd_even_transpose_sort_impl<i, i>
{
  template<typename RandomAccessIterator, typename Compare>
  static __device__ void sort(RandomAccessIterator, int, Compare) { }
};


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename Compare>
__forceinline__ __device__
void stable_odd_even_transpose_sort(const bounded<bound,agent<grainsize> > &,
                                    RandomAccessIterator first, RandomAccessIterator last,
                                    Compare comp)
{
  stable_odd_even_transpose_sort_impl<0, bound>::sort(first, last - first, comp);
} // end stable_odd_even_transpose_sort()


} // end sort_detail
} // end detail


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
__forceinline__ __device__
void stable_sort_by_key(const bounded<bound,agent<grainsize> > &exec,
                        RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        Compare comp)
{
  bulk::detail::sort_detail::stable_odd_even_transpose_sort_by_key(exec, keys_first, keys_last, values_first, comp);
} // end stable_sort_by_key()


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename Compare>
__forceinline__ __device__
void stable_sort(const bounded<bound,agent<grainsize> > &exec,
                 RandomAccessIterator first, RandomAccessIterator last,
                 Compare comp)
{
  bulk::detail::sort_detail::stable_odd_even_transpose_sort(exec, first, last, comp);
} // end stable_sort()


} // end bulk
BULK_NS_SUFFIX

