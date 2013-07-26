#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_policy.hpp>

BULK_NS_PREFIX
namespace bulk
{


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__forceinline__ __device__
void scatter_if(const bounded_agent<bound,agent<grainsize> > &exec,
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


template<std::size_t bound,
         std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1, 
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize
>::type
scatter_if(bulk::bounded_agent<
             bound,
             bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
           > &g,
           RandomAccessIterator1 first,
           RandomAccessIterator1 last,
           RandomAccessIterator2 map,
           RandomAccessIterator3 stencil,
           RandomAccessIterator4 result)
{
  typedef typename bulk::bounded_agent<
    bound,
    bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
  >::size_type size_type;

  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::agent_type agent_type;

  size_type n = last - first;

  size_type tid = g.this_exec.index();

  // avoid branches when possible
  if(n == bound)
  {
    for(size_type i = 0; i < g.this_exec.grainsize(); ++i)
    {
      size_type idx = g.size() * i + tid;

      if(stencil[idx])
      {
        result[map[idx]] = first[idx];
      } // end if
    } // end for
  } // end if
  else if(n < bound)
  {
    for(size_type i = 0; i < g.this_exec.grainsize(); ++i)
    {
      size_type idx = g.size() * i + tid;

      if(idx < (last - first) && stencil[idx])
      {
        result[map[idx]] = first[idx];
      } // end if
    } // end for
  } // end if

  g.wait();
} // end scatter_if()


template<std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1, 
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__device__
void scatter_if(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
                RandomAccessIterator1 first,
                RandomAccessIterator1 last,
                RandomAccessIterator2 map,
                RandomAccessIterator3 stencil,
                RandomAccessIterator4 result)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

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
  else if(n < chunk_size)
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type idx = g.size() * i + tid;

      if(idx < (last - first) && stencil[idx])
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

  g.wait();
} // end scatter_if


} // end bulk
BULK_NS_SUFFIX

