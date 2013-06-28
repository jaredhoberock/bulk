#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_group.hpp>
#include <bulk/sequential_executor.hpp>

BULK_NS_PREFIX
namespace bulk
{


template<std::size_t bound,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__forceinline__ __device__
void scatter_if(const bounded_executor<bound> &exec,
                RandomAccessIterator1 first,
                RandomAccessIterator1 last,
                RandomAccessIterator2 map,
                RandomAccessIterator3 stencil,
                RandomAccessIterator4 result)
{
  typedef int size_type;

  size_type n = last - first;

  for(size_type i = 0; i < bound; ++i)
  {
    if(i < n && stencil[i])
    {
      result[map[i]] = first[i];
    } // end if
  } // end for
} // end scatter_if()


template<std::size_t groupsize,
         std::size_t grainsize_,
         typename RandomAccessIterator1, 
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__device__
void scatter_if(bulk::static_execution_group<groupsize,grainsize_> &g,
                RandomAccessIterator1 first,
                RandomAccessIterator1 last,
                RandomAccessIterator2 map,
                RandomAccessIterator3 stencil,
                RandomAccessIterator4 result)
{
  typedef int size_type;

  const size_type grainsize = bulk::static_execution_group<groupsize,grainsize_>::static_grainsize;

  size_type chunk_size = g.size() * grainsize;

  size_type n = last - first;

  size_type tid = g.this_exec.index();

  // important special case which avoids the expensive for loop below
  if(chunk_size == n)
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type idx = g.size() * i + tid;

      if(stencil[idx])
      {
        result[map[idx]] = first[idx];
      } // end if
    } // end for
  } // end if
  else
  {
    for(;
        first < last;
        first += chunk_size, map += chunk_size, stencil += chunk_size)
    {
      if((last - first) >= chunk_size)
      {
        // avoid conditional accesses when possible
        for(size_type i = 0; i < grainsize; ++i)
        {
          size_type idx = g.size() * i + tid;

          if(stencil[idx])
          {
            result[map[idx]] = first[idx];
          } // end if
        } // end for
      } // end if
      else
      {
        for(size_type i = 0; i < grainsize; ++i)
        {
          size_type idx = g.size() * i + tid;

          if(idx < (last - first) && stencil[idx])
          {
            result[map[idx]] = first[idx];
          } // end if
        } // end for
      } // end else
    } // end for
  } // end else
} // end scatter_if


} // end bulk
BULK_NS_SUFFIX

