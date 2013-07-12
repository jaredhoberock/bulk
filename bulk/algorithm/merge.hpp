#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/sequential_executor.hpp>
#include <bulk/malloc.hpp>
#include <bulk/algorithm/copy.hpp>
#include <bulk/uninitialized.hpp>


BULK_NS_PREFIX
namespace bulk
{


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename Compare>
__device__
Size merge_path(RandomAccessIterator1 first1, Size n1,
                RandomAccessIterator2 first2, Size n2,
                Size diag,
                Compare comp)
{
  Size begin = thrust::max<Size>(Size(0), diag - n2);
  Size end = thrust::min<Size>(diag, n1);
  
  while(begin < end)
  {
    Size mid = (begin + end) >> 1;

    if(comp(first2[diag - 1 - mid], first1[mid]))
    {
      end = mid;
    } // end if
    else
    {
      begin = mid + 1;
    } // end else
  } // end while

  return begin;
} // end merge_path()


template<std::size_t bound,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Compare>
__device__
OutputIterator merge(const bulk::bounded_executor<bound> &e,
                     InputIterator1 first1, InputIterator1 last1,
                     InputIterator2 first2, InputIterator2 last2,
                     OutputIterator result,
                     Compare comp)
{
  typedef int size_type;

  typedef typename thrust::iterator_value<InputIterator1>::type value_type1;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type2;

  size_type n1 = last1 - first1;
  size_type idx1 = 0;

  size_type n2 = last2 - first2;
  size_type idx2 = 0;

  uninitialized<value_type1> a;
  uninitialized<value_type2> b;

  if(n1)
  {
    a.construct(first1[0]);
  } // end if

  if(n2)
  {
    b.construct(first2[0]);
  } // end if

  size_type i = 0;
  #pragma unroll
  for(; i < bound; ++i)
  {
    // 4 cases:
    // 0. both ranges are exhausted
    // 1. range 1 is exhausted
    // 2. range 2 is exhausted
    // 3. neither range is exhausted

    const bool exhausted1 = idx1 >= n1;
    const bool exhausted2 = idx2 >= n2;

    if(exhausted1 && exhausted2)
    {
      break;
    } // end if
    else if(exhausted1)
    {
      result[i] = b;
      ++idx2;
    } // end else if
    else if(exhausted2)
    {
      result[i] = a;
      ++idx1;
    } // end else if
    else
    {
      if(!comp(b.get(),a.get()))
      {
        result[i] = a;
        ++idx1;

        if(idx1 < n1)
        {
          a = first1[idx1];
        } // end if
      } // end if
      else
      {
        result[i] = b;
        ++idx2;

        if(idx2 < n2)
        {
          b = first2[idx2];
        } // end if
      } // end else
    } // end else
  } // end for i

  if(n1)
  {
    a.destroy();
  } // end if

  if(n2)
  {
    b.destroy();
  } // end if

  return result + i;
} // end merge


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename Compare>
__device__
typename thrust::detail::enable_if<
  (bound <= groupsize * grainsize)
>::type
inplace_merge(const bulk::bounded_static_execution_group<bound,groupsize,grainsize> &g_,
              RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last,
              Compare comp)
{
  bulk::bounded_static_execution_group<bound,groupsize,grainsize> &g =
    const_cast<bulk::bounded_static_execution_group<bound,groupsize,grainsize>&>(g_);

  typedef int size_type;

  size_type n1 = middle - first;
  size_type n2 = last - middle;

  // find the start of each local merge
  size_type local_offset = grainsize * g.this_exec.index();

  size_type mp = bulk::merge_path(first, n1, middle, n2, local_offset, comp);
  
  // do a local sequential merge
  size_type local_offset1 = mp;
  size_type local_offset2 = n1 + local_offset - mp;
  
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  value_type local_result[grainsize];
  bulk::merge(bulk::bound<grainsize>(g.this_exec),
              first + local_offset1, middle,
              first + local_offset2, last,
              local_result,
              comp);

  g.wait();

  // copy local result back to source
  // this is faster than getting the size from merge's result
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n1 + n2 - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_result, local_size, first + local_offset); 

  g.wait();
} // end inplace_merge()


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename Compare>
__device__
typename thrust::detail::enable_if<
  (bound <= groupsize * grainsize),
  RandomAccessIterator3
>::type
merge(const bulk::bounded_static_execution_group<bound,groupsize,grainsize> &g_,
      RandomAccessIterator1 first1, RandomAccessIterator1 last1,
      RandomAccessIterator2 first2, RandomAccessIterator2 last2,
      RandomAccessIterator3 result,
      Compare comp)
{
  bulk::bounded_static_execution_group<bound,groupsize,grainsize> &g =
    const_cast<bulk::bounded_static_execution_group<bound,groupsize,grainsize>&>(g_);

  typedef int size_type;

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // find the start of each local merge
  size_type local_offset = grainsize * g.this_exec.index();

  size_type mp = bulk::merge_path(first1, n1, first2, n2, local_offset, comp);
  
  // do a local sequential merge
  size_type local_offset1 = mp;
  size_type local_offset2 = local_offset - mp;
  
  typedef typename thrust::iterator_value<RandomAccessIterator3>::type value_type;
  value_type local_result[grainsize];
  bulk::merge(bulk::bound<grainsize>(g.this_exec),
              first1 + local_offset1, last1,
              first2 + local_offset2, last2,
              local_result,
              comp);

  // store local result
  // this is faster than getting the size from merge's result
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n1 + n2 - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_result, local_size, result + local_offset); 

  g.wait();

  return result + thrust::min<size_type>(groupsize * grainsize, n1 + n2);
} // end merge()


namespace detail
{
namespace merge_detail
{


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
__device__
RandomAccessIterator4
  bounded_merge_with_buffer(bulk::static_execution_group<groupsize,grainsize> &exec,
                            RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                            RandomAccessIterator3 buffer,
                            RandomAccessIterator4 result,
                            Compare comp)
{
  typedef int size_type;

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // copy into the buffer
  bulk::copy_n(bulk::bound<groupsize * grainsize>(exec),
               make_join_iterator(first1, n1, first2),
               n1 + n2,
               buffer);

  // inplace merge in the buffer
  bulk::inplace_merge(bulk::bound<groupsize * grainsize>(exec),
                      buffer, buffer + n1, buffer + n1 + n2,
                      comp);
  
  // copy to the result
  // XXX this might be slightly faster with a bounded copy_n
  return bulk::copy_n(exec, buffer, n1 + n2, result);
} // end bounded_merge_with_buffer()


} // end merge_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__device__
RandomAccessIterator3 merge(bulk::static_execution_group<groupsize,grainsize> &exec,
                            RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  typedef int size_type;

  typedef typename thrust::iterator_value<RandomAccessIterator3>::type value_type;

  value_type *buffer = reinterpret_cast<value_type*>(bulk::malloc(exec, exec.size() * exec.grainsize() * sizeof(value_type)));

  size_type chunk_size = exec.size() * exec.grainsize();

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // avoid the search & loop when possible
  if(n1 + n2 <= chunk_size)
  {
    result = detail::merge_detail::bounded_merge_with_buffer(exec, first1, last1, first2, last2, buffer, result, comp);
  } // end if
  else
  {
    while((first1 < last1) || (first2 < last2))
    {
      size_type n1 = last1 - first1;
      size_type n2 = last2 - first2;

      size_type diag = thrust::min<size_type>(chunk_size, n1 + n2);

      size_type mp = bulk::merge_path(first1, n1, first2, n2, diag, comp);

      result = detail::merge_detail::bounded_merge_with_buffer(exec,
                                                               first1, first1 + mp,
                                                               first2, first2 + diag - mp,
                                                               buffer,
                                                               result,
                                                               comp);

      first1 += mp;
      first2 += diag - mp;
    } // end while
  } // end else

  bulk::free(exec, buffer);

  return result;
} // end merge()


} // end bulk
BULK_NS_SUFFIX

