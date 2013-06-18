#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <bulk/bulk.hpp>
#include "join_iterator.hpp"
#include "time_invocation_cuda.hpp"


template<std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator,
         typename Compare>
__device__
void bounded_inplace_merge(bulk::static_execution_group<groupsize,grainsize> &g,
                           RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp)
{
  int n1 = middle - first;
  int n2 = last - middle;

  // Run a merge path to find the start of the serial merge for each thread.
  int diag = grainsize * threadIdx.x;

  // XXX could invent an "inplace_merge_path" variant which didn't require redundant parameters n1 & middle
  int mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(first, n1, middle, n2, diag, comp);
  
  // Compute the ranges of the sources in shared memory.
  int local_offset1 = mp;
  int local_offset2 = n1 + diag - mp;
  
  // Serial merge into register.
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  value_type local_result[grainsize];
  bulk::merge(bulk::bound<grainsize>(g.this_exec),
              first + local_offset1, middle,
              first + local_offset2, last,
              local_result,
              comp);

  g.wait();

  // local result back to source
  int local_offset = grainsize * threadIdx.x;

  // this is faster than getting the size from merge's result
  int local_size = thrust::max<int>(0, thrust::min<int>(grainsize, n1 + n2 - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_result, local_size, first + local_offset); 

  g.wait();
}


// XXX this is essentially a bounded version for group copy_n
//     the bound is groupsize * grainsize
template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
__device__
RandomAccessIterator2 bounded_copy_n(bulk::static_execution_group<groupsize,grainsize> &g,
                                     RandomAccessIterator1 first,
                                     Size n,
                                     RandomAccessIterator2 result)
{
  typedef int size_type;

  size_type tid = g.this_exec.index();

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  // XXX make this an uninitialized array
  value_type stage[grainsize];

  // avoid conditional accesses when possible
  if(groupsize * grainsize <= n)
  {
    #pragma unroll
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type src_idx = g.size() * i + tid;
      stage[i] = first[src_idx];
    } // end for i

    #pragma unroll
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type dst_idx = g.size() * i + tid;
      result[dst_idx] = stage[i];
    } // end for i
  } // end if
  else
  {
    #pragma unroll
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type src_idx = g.size() * i + tid;
      if(src_idx < n)
      {
        stage[i] = first[src_idx];
      } // end if
    } // end for

    #pragma unroll
    for(size_type i = 0; i < grainsize; ++i)
    {
      size_type dst_idx = g.size() * i + tid;
      if(dst_idx < n)
      {
        result[dst_idx] = stage[i];
      } // end if
    } // end for
  } // end else

  g.wait();

  return result + thrust::min<Size>(g.size() * grainsize, n);
}


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__device__
RandomAccessIterator3
  bounded_merge(bulk::static_execution_group<groupsize,grainsize> &exec,
                RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                RandomAccessIterator3 result,
                Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator3>::type value_type;

  __shared__ value_type buffer[groupsize * grainsize];

  typedef int size_type;

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // copy into the buffer
  bounded_copy_n(exec,
                 make_join_iterator(first1, n1, first2),
                 n1 + n2,
                 buffer);

  // inplace merge in the buffer
  bounded_inplace_merge(exec, buffer, buffer + n1, buffer + n1 + n2, comp);
  
  // copy to the result
  // XXX this might be slightly faster with a bounded_copy_n
  return bulk::copy_n(exec, buffer, n1 + n2, result);
}


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__device__
RandomAccessIterator3 merge(bulk::static_execution_group<groupsize,grainsize> &exec,
                            RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  typedef int size_type;

  size_type chunk_size = exec.size() * exec.grainsize();

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // avoid the search & loop when possible
  if(n1 + n2 <= chunk_size)
  {
    result = bounded_merge(exec, first1, last1, first2, last2, result, comp);
  } // end if
  else
  {
    while((first1 < last1) || (first2 < last2))
    {
      size_type n1 = last1 - first1;
      size_type n2 = last2 - first2;

      size_type diag = thrust::min<size_type>(chunk_size, n1 + n2);

      size_type mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(first1, n1, first2, n2, diag, comp);

      result = bounded_merge(exec,
                             first1, first1 + mp,
                             first2, first2 + diag - mp,
                             result,
                             comp);

      first1 += mp;
      first2 += diag - mp;
    } // end while
  } // end else

  return result;
} // end merge()


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
__global__
void merge_n(RandomAccessIterator1 first1, Size n1,
             RandomAccessIterator2 first2, Size n2,
             RandomAccessIterator3 merge_paths_first,
             RandomAccessIterator4 result,
             Compare comp)
{
  bulk::static_execution_group<groupsize,grainsize> g;

  typedef int size_type;

  size_type elements_per_group = g.size() * g.grainsize();

  size_type mp0  = merge_paths_first[g.index()];
  size_type mp1  = merge_paths_first[g.index()+1];
  size_type diag = elements_per_group * g.index();
  
  bounded_merge(g,
                first1 + mp0,        first1 + mp1,
                first2 + diag - mp0, first2 + thrust::min<Size>(n1 + n2, diag + elements_per_group) - mp1, // <- surely that can be simplified
                result + elements_per_group * g.index(),
                comp);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 my_merge(RandomAccessIterator1 first1,
                               RandomAccessIterator1 last1,
                               RandomAccessIterator2 first2,
                               RandomAccessIterator2 last2,
                               RandomAccessIterator3 result,
                               Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);

  // XXX these seem to work well for K20c but could use some comprehensive tuning
  const int NT = 128 + 64;
  const int VT = 9;

  typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
  int2 launch = Tuning::GetLaunchParams(*ctx);
  
  const int NV = launch.x * launch.y;

  // find partitions
  MGPU_MEM(int) partitionsDevice =
    mgpu::MergePathPartitions<mgpu::MgpuBoundsLower>(
      first1, last1 - first1,
      first2, last2 - first2,
      NV,
      0,
      comp,
      *ctx);

  // merge partitions
  int n = (last1 - first1) + (last2 - first2);

  // XXX it's easy to launch too many blocks this way
  //     we need to cap it and virtualize
  int num_blocks = (n + NV - 1) / NV;

  merge_n<NT,VT><<<num_blocks, launch.x>>>(first1, last1 - first1, first2, last2 - first2, partitionsDevice->get(), result, comp);

  return result + n;
} // end merge()


template<typename T>
void my_merge(const thrust::device_vector<T> *a,
              const thrust::device_vector<T> *b,
              thrust::device_vector<T> *c)
{
  my_merge(a->begin(), a->end(),
           b->begin(), b->end(),
           c->begin(),
           thrust::less<T>());
}


template<typename T>
void sean_merge(const thrust::device_vector<T> *a,
                const thrust::device_vector<T> *b,
                thrust::device_vector<T> *c)
{
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);
  mgpu::MergeKeys(a->begin(), a->size(),
                  b->begin(), b->size(),
                  c->begin(),
                  thrust::less<T>(),
                  *ctx);
}


template<typename T>
void thrust_merge(const thrust::device_vector<T> *a,
                  const thrust::device_vector<T> *b,
                  thrust::device_vector<T> *c)
{
  thrust::merge(a->begin(), a->end(),
                b->begin(), b->end(),
                c->begin(),
                thrust::less<T>());
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
    return x;
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
  thrust::device_vector<T> a(n / 2), b(n / 2);
  thrust::device_vector<T> c(n);

  random_fill(a);
  random_fill(b);

  thrust::sort(a.begin(), a.end());
  thrust::sort(b.begin(), b.end());

  my_merge(&a, &b, &c);
  double my_msecs = time_invocation_cuda(50, my_merge<T>, &a, &b, &c);

  sean_merge(&a, &b, &c);
  double sean_msecs = time_invocation_cuda(50, sean_merge<T>, &a, &b, &c);

  thrust_merge(&a, &b, &c);
  double thrust_msecs = time_invocation_cuda(50, thrust_merge<T>, &a, &b, &c);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;

  std::cout << "Performance relative to Sean: " << sean_msecs / my_msecs << std::endl;
  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;
}


template<typename T>
void validate(size_t n)
{
  thrust::device_vector<T> a(n / 2), b(n / 2);
  thrust::device_vector<T> c(n);

  random_fill(a);
  random_fill(b);

  thrust::sort(a.begin(), a.end());
  thrust::sort(b.begin(), b.end());

  thrust::device_vector<T> ref(n);
  thrust::merge(a.begin(), a.end(), b.begin(), b.end(), ref.begin());

  my_merge(&a, &b, &c);

  std::cout << "CUDA error: " << cudaGetErrorString(cudaThreadSynchronize()) << std::endl;

  assert(c == ref);
}


int main()
{
  size_t n = 123456789;

  validate<int>(n);

  std::cout << "Large input: " << std::endl;
  std::cout << "int: " << std::endl;
  compare<int>(n);

  std::cout << "float: " << std::endl;
  compare<float>(n);

  std::cout << "double: " << std::endl;
  compare<double>(n);
  std::cout << std::endl;

  return 0;
}

