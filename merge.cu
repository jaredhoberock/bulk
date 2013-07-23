#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/detail/temporary_array.h>
#include <bulk/bulk.hpp>
#include "join_iterator.hpp"
#include "time_invocation_cuda.hpp"


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size,typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
__device__
RandomAccessIterator4
  staged_merge(bulk::concurrent_group<bulk::sequential_executor<grainsize>,groupsize> &exec,
               RandomAccessIterator1 first1, Size n1,
               RandomAccessIterator2 first2, Size n2,
               RandomAccessIterator3 stage,
               RandomAccessIterator4 result,
               Compare comp)
{
  // copy into the stage
  bulk::copy_n(bulk::bound<groupsize * grainsize>(exec),
               make_join_iterator(first1, n1, first2),
               n1 + n2,
               stage);

  // inplace merge in the stage
  bulk::inplace_merge(bulk::bound<groupsize * grainsize>(exec),
                      stage, stage + n1, stage + n1 + n2,
                      comp);
  
  // copy to the result
  // XXX this might be slightly faster with a bounded copy_n
  return bulk::copy_n(exec, stage, n1 + n2, result);
} // end staged_merge()


struct merge_kernel
{
  template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
  __device__
  void operator()(bulk::concurrent_group<bulk::sequential_executor<grainsize>,groupsize> &g,
                  RandomAccessIterator1 first1, Size n1,
                  RandomAccessIterator2 first2, Size n2,
                  RandomAccessIterator3 merge_paths_first,
                  RandomAccessIterator4 result,
                  Compare comp)
  {
    typedef int size_type;

    size_type elements_per_group = g.size() * g.this_exec.grainsize();

    // determine the ranges to merge
    size_type mp0  = merge_paths_first[g.index()];
    size_type mp1  = merge_paths_first[g.index()+1];
    size_type diag = elements_per_group * g.index();

    size_type local_size1 = mp1 - mp0;
    size_type local_size2 = thrust::min<size_type>(n1 + n2, diag + elements_per_group) - mp1 - diag + mp0;

    first1 += mp0;
    first2 += diag - mp0;
    result += elements_per_group * g.index();

    typedef typename thrust::iterator_value<RandomAccessIterator4>::type value_type;

#if __CUDA_ARCH__ >= 200
    // merge through a stage
    value_type *stage = reinterpret_cast<value_type*>(bulk::malloc(g, elements_per_group * sizeof(value_type)));

    if(bulk::is_on_chip(stage))
    {
      staged_merge(g,
                   first1, local_size1,
                   first2, local_size2,
                   bulk::on_chip_cast(stage),
                   result,
                   comp);
    } // end if
    else
    {
      staged_merge(g,
                   first1, local_size1,
                   first2, local_size2,
                   stage,
                   result,
                   comp);
    } // end else

    bulk::free(g, stage);
#else
    bulk::uninitialized_array<value_type, groupsize * grainsize> stage;
    staged_merge(g, first1, local_size1, first2, local_size2, stage.data(), result, comp);
#endif
  } // end operator()
}; // end merge_kernel


template<typename Size, typename RandomAccessIterator1,typename RandomAccessIterator2, typename Compare>
struct locate_merge_path
{
  Size partition_size;
  RandomAccessIterator1 first1, last1;
  RandomAccessIterator2 first2, last2;
  Compare comp;

  locate_merge_path(Size partition_size, RandomAccessIterator1 first1, RandomAccessIterator1 last1, RandomAccessIterator2 first2, RandomAccessIterator2 last2, Compare comp)
    : partition_size(partition_size),
      first1(first1), last1(last1),
      first2(first2), last2(last2),
      comp(comp)
  {}

  template<typename Index>
  __device__
  Size operator()(Index i)
  {
    Size n1 = last1 - first1;
    Size n2 = last2 - first2;
    Size diag = thrust::min<Size>(partition_size * i, n1 + n2);
    return bulk::merge_path(first1, n1, first2, n2, diag, comp);
  }
};


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
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference_type;
  typedef int size_type;

  // 90/86/97
  const size_type groupsize = (sizeof(value_type) == sizeof(int)) ? 256 : 256 + 32;
  const size_type grainsize = (sizeof(value_type) == sizeof(int)) ? 9   : 5;
  
  const size_type tile_size = groupsize * grainsize;

  // XXX it's easy to launch too many blocks this way
  //     we need to cap it and virtualize
  difference_type n = (last1 - first1) + (last2 - first2);

  difference_type num_groups = (n + tile_size - 1) / tile_size;

  thrust::cuda::tag t;
  thrust::detail::temporary_array<size_type,thrust::cuda::tag> merge_paths(t, num_groups + 1);

  thrust::tabulate(merge_paths.begin(), merge_paths.end(), locate_merge_path<size_type,RandomAccessIterator1,RandomAccessIterator2,Compare>(tile_size,first1,last1,first2,last2,comp));

  // merge partitions
  size_type heap_size = tile_size * sizeof(value_type);
  bulk::concurrent_group<bulk::sequential_executor<grainsize>,groupsize> g(heap_size);
  bulk::async(bulk::par(g, num_groups), merge_kernel(), bulk::root.this_exec, first1, last1 - first1, first2, last2 - first2, merge_paths.begin(), result, comp);

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

