#pragma once

#include <bulk/thread_group.hpp>
#include <bulk/malloc.hpp>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace bulk
{
namespace detail
{
namespace scan_detail
{


template<unsigned int grainsize, typename Iterator1, typename difference_type, typename Iterator2>
__device__
void copy_n_with_grainsize(Iterator1 first, difference_type n, Iterator2 result)
{
  for(Iterator1 last = first + n;
      first < last;
      first += grainsize, result += grainsize)
  {
    for(int i = 0; i < grainsize; ++i)
    {
      if(i < (last - first))
      {
        result[i] = first[i];
      }
    }
  }
}


template<typename ThreadGroup, typename Iterator, typename difference_type, typename BinaryFunction>
__device__ void small_inclusive_scan_n(ThreadGroup &g, Iterator first, difference_type n, BinaryFunction binary_op)
{
  typedef typename ThreadGroup::size_type size_type;

  T x;

  size_type tid = g.this_thread.index();

  if(tid < n)
  {
    x = first[tid];
  }

  g.wait();

  for(size_type offset = 1; offset < n; offset += offset)
  {
    if(tid >= offset && tid < n)
    {
      x = binary_op(first[tid - offset], x);
    }

    g.wait();

    if(tid < n)
    {
      first[tid] = x;
    }

    g.wait();
  }
}


template<typename ThreadGroup, typename Iterator, typename difference_type, typename T, typename BinaryFunction>
__device__ T small_exclusive_scan_n(ThreadGroup &g, Iterator first, difference_type n, T init, BinaryFunction binary_op)
{
  typedef typename ThreadGroup::size_type size_type;

  T x;

  size_type tid = g.this_thread.index();

  if(n > 0 && tid == 0)
  {
    *first = binary_op(init, *first);
  }

  g.wait();

  if(tid < n)
  {
    x = first[tid];
  }

  g.wait();

  inclusive_scan_n(g, first, n, binary_op);

  T result = n > 0 ? first[n - 1] : init;

  x = (tid == 0 || tid - 1 >= n) ? init : first[tid - 1];

  g.wait();

  if(tid < n)
  {
    first[tid] = x;
  }

  g.wait();

  return result;
}


} // end scan_detail
} // end detail


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BinaryFunction>
__device__
RandomAccessIterator2 inclusive_scan(static_thread_group<size,grainsize> &this_group,
                                     RandomAccessIterator1 first,
                                     RandomAccessIterator1 last,
                                     RandomAccessIterator2 result,
                                     BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference_type;
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type input_type;
  
  // XXX this needs to be inferred from the iterators and binary_op
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type intermediate_type;
  
  typedef typename bulk::static_thread_group<size,grainsize>::size_type size_type;
  
  const difference_type n = last - first;
  const size_type elements_per_group = size * grainsize;
  
  // we don't need the inputs and the results at the same time
  // so we can overlay these arrays
  union stage
  {
    input_type *inputs;
    intermediate_type *results;
  };
  
  stage s_stage;
  s_stage.inputs = reinterpret_cast<input_type*>(bulk::malloc(this_group, elements_per_group * thrust::max<int>(sizeof(input_type), sizeof(intermediate_type))));
  
  intermediate_type *s_sums = reinterpret_cast<intermediate_type*>(bulk::malloc(this_group, size * sizeof(intermediate_type)));
  
  size_type tid = this_group.this_thread.index();
  
  // carry is the sum over all previous iterations
  intermediate_type carry = first[0];
  
  if(this_group.this_thread.index() == 0)
  {
    result[0] = carry;
  }
  
  for(difference_type start = 1; start < n; start += elements_per_group)
  {
    difference_type partition_size = thrust::min<difference_type>(elements_per_group, n - start);
  
    // stage data through shared memory
    bulk::copy_n(this_group, first + start, partition_size, s_stage.inputs);
    
    // Transpose data into register in thread order. Reduce terms serially.
    // XXX this should probably be uninitialized_array<input_type>
    input_type local_inputs[grainsize];
  
    size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, partition_size - grainsize * tid));
  
    size_type local_offset = grainsize * tid;
  
    // XXX this should probably be uninitialized<intermediate_type>
    intermediate_type x;
  
    if(local_size > 0)
    {
      // XXX would be cool simply to call
      // bulk::copy_n(this_group.this_thread, ...) instead
      copy_n_with_grainsize<grainsize>(s_stage.inputs + local_offset, local_size, local_inputs);
  
      // XXX this should actually be accumulate because we desire non-commutativity
      x = thrust::reduce(thrust::seq, local_inputs + 1, local_inputs + local_size, local_inputs[0], binary_op);
  
      s_sums[tid] = x;
    }
  
    this_group.wait();
  
    // scan this group's sums
    // XXX is this really the correct number of sums?
    //     it should be divide_ri(partition_size, grainsize)
    carry = scan_detail::small_exclusive_scan_n(this_group, s_sums, thrust::min<size_type>(size,partition_size), carry, binary_op);
  
    // each thread does an inplace scan locally while incorporating the carries
    if(local_size > 0)
    {
      local_inputs[0] = binary_op(s_sums[tid],local_inputs[0]);
  
      // XXX would be cool simply to call
      // bulk::inclusive_scan(this_group.this_thread, ...) instead
      thrust::inclusive_scan(thrust::seq, local_inputs, local_inputs + local_size, local_inputs, binary_op);
  
      // XXX would be cool simply to call
      // bulk::copy_n(this_group.this_thread, ...) instead
      copy_n_with_grainsize<grainsize>(local_inputs, local_size, s_stage.results + local_offset);
    }
  
    this_group.wait();
    
    // store results
    bulk::copy_n(this_group, s_stage.results, partition_size, result + start);
  }
  
  bulk::free(this_group, s_stage.inputs);
  bulk::free(this_group, s_sums);

  return last;
} // end inclusive_scan()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
RandomAccessIterator2 exclusive_scan(static_thread_group<size,grainsize> &this_group,
                                     RandomAccessIterator1 first,
                                     RandomAccessIterator1 last,
                                     RandomAccessIterator2 result,
                                     T init,
                                     BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type input_type;
  
  // XXX this needs to be inferred from the iterators and binary_op
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type intermediate_type;
  
  typedef typename bulk::static_thread_group<size,grainsize>::size_type size_type;
  
  const difference_type n = last - first;
  const size_type elements_per_group = size * grainsize;
  
  // we don't need the inputs and the results at the same time
  // so we can overlay these arrays
  union stage
  {
    input_type *inputs;
    intermediate_type *results;
  };
  
  stage s_stage;
  s_stage.inputs = reinterpret_cast<input_type*>(bulk::malloc(this_group, elements_per_group * thrust::max<int>(sizeof(input_type), sizeof(intermediate_type))));
  
  intermediate_type *s_sums = reinterpret_cast<intermediate_type*>(bulk::malloc(this_group, size * sizeof(intermediate_type)));
  
  size_type tid = this_group.this_thread.index();
  
  // carry is the sum over all previous iterations
  intermediate_type carry = init;
  
  if(this_group.this_thread.index() == 0)
  {
    result[0] = carry;
  }

  ++first;
  
  for(difference_type start = 1; start < n; start += elements_per_group)
  {
    difference_type partition_size = thrust::min<difference_type>(elements_per_group, n - start);
  
    // stage data through shared memory
    bulk::copy_n(this_group, first + start, partition_size, s_stage.inputs);
    
    // Transpose data into register in thread order. Reduce terms serially.
    // XXX this should probably be uninitialized_array<input_type>
    input_type local_inputs[grainsize];
  
    size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, partition_size - grainsize * tid));
  
    size_type local_offset = grainsize * tid;
  
    // XXX this should probably be uninitialized<intermediate_type>
    intermediate_type x;
  
    if(local_size > 0)
    {
      // XXX would be cool simply to call
      // bulk::copy_n(this_group.this_thread, ...) instead
      copy_n_with_grainsize<grainsize>(s_stage.inputs + local_offset, local_size, local_inputs);
  
      // XXX this should actually be accumulate because we desire non-commutativity
      x = thrust::reduce(thrust::seq, local_inputs + 1, local_inputs + local_size, local_inputs[0], binary_op);
  
      s_sums[tid] = x;
    }
  
    this_group.wait();
  
    // scan this group's sums
    // XXX is this really the correct number of sums?
    //     it should be divide_ri(partition_size, grainsize)
    carry = scan_detail::small_exclusive_scan_n(this_group, s_sums, thrust::min<size_type>(size,partition_size), carry, binary_op);
  
    // each thread does an inplace scan locally while incorporating the carries
    if(local_size > 0)
    {
      // XXX would be cool simply to call
      // bulk::exclusive_scan(this_group.this_thread, ...) instead
      thrust::exclusive_scan(thrust::seq, local_inputs, local_inputs + local_size, local_inputs, s_sums[tid], binary_op);
  
      // XXX would be cool simply to call
      // bulk::copy_n(this_group.this_thread, ...) instead
      copy_n_with_grainsize<grainsize>(local_inputs, local_size, s_stage.results + local_offset);
    }
  
    this_group.wait();
    
    // store results
    bulk::copy_n(this_group, s_stage.results, partition_size, result + start);
  }
  
  bulk::free(this_group, s_stage.inputs);
  bulk::free(this_group, s_sums);

  return last;
} // end exclusive_scan()


} // end bulk

