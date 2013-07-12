#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/algorithm/reduce.hpp>
#include <bulk/sequential_executor.hpp>
#include <bulk/uninitialized.hpp>
#include <thrust/detail/type_traits/function_traits.h>

BULK_NS_PREFIX
namespace bulk
{


template<std::size_t bound,
         typename RandomAccessIterator,
         typename T,
         typename BinaryFunction>
__forceinline__ __device__
T accumulate(const bounded_executor<bound> &exec,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  typedef typename bounded_executor<bound>::size_type size_type;
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  size_type n = last - first;

  #pragma unroll
  for(size_type i = 0; i < exec.bound(); ++i)
  {
    if(i < n)
    {
      init = binary_op(init, first[i]);
    } // end if
  } // end for i

  return init;
} // end accumulate()


namespace detail
{
namespace accumulate_detail
{


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T>
struct buffer
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  union
  {
    value_type inputs[groupsize * grainsize];
    T sums[groupsize];
  }; // end union
}; // end buffer


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T accumulate(bulk::static_execution_group<groupsize,grainsize> &g,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  typedef int size_type;

  const size_type elements_per_group = groupsize * grainsize;

  size_type tid = g.this_exec.index();

  T sum = init;

  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  typedef detail::accumulate_detail::buffer<
    groupsize,
    grainsize,
    RandomAccessIterator,
    T
  > buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));
#else
  __shared__ uninitialized<buffer_type> buffer_impl;
  buffer_type *buffer = &buffer_impl.get();
#endif
  
  for(; first < last; first += elements_per_group)
  {
    // XXX each iteration is essentially a bounded accumulate
    
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // copy partition into smem
    bulk::copy_n(g, first, partition_size, buffer->inputs);
    
    T this_sum;
    size_type local_offset = grainsize * g.this_exec.index();

    size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, partition_size - grainsize * tid));

    if(local_size)
    {
      this_sum = buffer->inputs[local_offset];
      this_sum = bulk::accumulate(bound<grainsize-1>(g.this_exec),
                                  buffer->inputs + local_offset + 1,
                                  buffer->inputs + local_offset + local_size,
                                  this_sum,
                                  binary_op);
    } // end if

    g.wait();

    if(local_size)
    {
      buffer->sums[tid] = this_sum;
    } // end if

    g.wait();
    
    // sum over the group
    sum = detail::reduce_detail::destructive_reduce_n(g, buffer->sums, thrust::min<size_type>(groupsize,n), sum, binary_op);
  } // end for

#if __CUDA_ARCH__ >= 200
  bulk::free(g, buffer);
#endif

  return sum;
} // end accumulate
} // end accumulate_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__
T accumulate(bulk::static_execution_group<groupsize,grainsize> &g,
             RandomAccessIterator first,
             RandomAccessIterator last,
             T init,
             BinaryFunction binary_op)
{
  // use reduce when the operator is commutative
  if(thrust::detail::is_commutative<BinaryFunction>::value)
  {
    init = bulk::reduce(g, first, last, init, binary_op);
  } // end if
  else
  {
    init = detail::accumulate_detail::accumulate(g, first, last, init, binary_op);
  } // end else

  return init;
} // end accumulate()


} // end bulk
BULK_NS_SUFFIX

