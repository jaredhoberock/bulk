#include <iostream>
#include <moderngpu.cuh>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <bulk/bulk.hpp>
#include "time_invocation_cuda.hpp"


template<int VT, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__device__
void my_SerialMerge(RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                    RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                    RandomAccessIterator3 result,
                    Compare comp)
{ 
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  int n1 = last1 - first1;
  int idx1 = 0;

  int n2 = last2 - first2;
  int idx2 = 0;

  value_type1 a = *first1;
  value_type2 b = *first2;

  bool exhausted2 = idx2 >= n2;
  
  #pragma unroll
  for(int i = 0; i < VT; ++i)
  {
    //bool p = (idx2 >= n2) || ((idx1 < n1) && !comp(b, a));
    bool p = exhausted2 || ((idx1 < n1) && !comp(b, a));
    
    if(idx1 < n1 || !exhausted2)
    {
      result[i] = p ? a : b;
    }
    
    if(p)
    {
      ++idx1;

      // XXX making this a conditional load is pretty expensive for some reason
      a = first1[idx1];
    }
    else
    {
      ++idx2;

      if(idx2 >= n2) exhausted2 = true;

      // XXX making this a conditional load is pretty expensive for some reason
      b = first2[idx2];
    }
  }
}


template<std::size_t bound,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Compare>
__device__
OutputIterator bounded_merge(InputIterator1 first1, InputIterator1 last1,
                             InputIterator2 first2, InputIterator2 last2,
                             OutputIterator result,
                             Compare comp)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type1;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type2;

  int n1 = last1 - first1;
  int idx1 = 0;

  int n2 = last2 - first2;
  int idx2 = 0;

  // XXX these should be uninitialized
  value_type1 a;
  value_type2 b;

  if(n1)
  {
    a = first1[0];
  }

  if(n2)
  {
    b = first2[0];
  }

  #pragma unroll
  for(int i = 0; i < bound; ++i)
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
      ;
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
      if(!comp(b,a))
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

  return result;
} // end bounded_merge


template<std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator,
         typename Compare>
__device__
void bounded_inplace_merge(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp)
{
  bulk::static_execution_group<groupsize,grainsize> g;

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
  my_SerialMerge<grainsize>(first + local_offset1, middle,
                            first + local_offset2, last,
                            local_result,
                            comp);

  g.wait();

  // local result back to source
  int local_offset = grainsize * threadIdx.x;
  int local_size = thrust::min<int>(grainsize, n1 + n2 - local_offset);
  bulk::copy_n(bulk::bound<grainsize>(g), local_result, local_size, first + local_offset); 

  g.wait();
}


template<int NT, int VT, typename KeysIt1, typename KeysIt2, typename KeysIt3, typename KeyType, typename Comp>
__device__
void my_DeviceMerge(KeysIt1 aKeys_global,
                    KeysIt2 bKeys_global,
                    int tid, int block,
                    int4 range,
                    KeyType* keys_shared,
                    KeysIt3 keys_global,
                    Comp comp)
{
  // Load the data into shared memory.
  int aCount = range.y - range.x;
  int bCount = range.w - range.z;
  mgpu::DeviceLoad2ToShared<NT, VT, VT>(aKeys_global + range.x, aCount, bKeys_global + range.z, bCount, tid, keys_shared);

  bounded_inplace_merge<NT,VT>(keys_shared, keys_shared + aCount, keys_shared + aCount + bCount, comp);
  
  // Store merged keys to global memory.
  mgpu::DeviceSharedToGlobal<NT, VT>(aCount + bCount, keys_shared, tid, keys_global + NT * VT * block);
}


template<typename Tuning, bool HasValues, bool MergeSort, typename KeysIt1, 
	typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
	typename ValsIt3, typename Comp>
__global__
void my_KernelMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global, int aCount,
                    KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
                    const int* mp_global,
                    int coop,
                    KeysIt3 keys_global, ValsIt3 vals_global,
                    Comp comp)
{
  typedef MGPU_LAUNCH_PARAMS Params;
  typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;
  typedef typename std::iterator_traits<ValsIt1>::value_type ValType;
  
  const int NT = Params::NT;
  const int VT = Params::VT;
  union Shared {
  	KeyType keys[NT * (VT + 1)];
  };
  __shared__ Shared shared;
  
  int tid = threadIdx.x;
  int block = blockIdx.x;
  
  int4 range = mgpu::ComputeMergeRange(aCount, bCount, block, coop, NT * VT, mp_global);
  
  my_DeviceMerge<NT, VT>(aKeys_global,
                         bKeys_global,
                         tid,
                         block,
                         range,
                         shared.keys, 
                         keys_global,
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

  const int NT = 128;
  const int VT = 11;
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
  int num_blocks = (n + NV - 1) / NV;
  my_KernelMerge<Tuning, false, false><<<num_blocks, launch.x, 0, 0>>>
    (first1, (const int*)0, last1 - first1,
     first2, (const int*)0, last2 - first2, 
      partitionsDevice->get(), 0,
      result,
      (int*)0,
      comp);

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


int main()
{
  size_t n = 123456789;

  thrust::device_vector<int> a(n / 2), b(n / 2);
  thrust::device_vector<int> c(n);

  random_fill(a);
  random_fill(b);
  thrust::sort(a.begin(), a.end());
  thrust::sort(b.begin(), b.end());

  my_merge(&a, &b, &c);

  thrust::device_vector<int> ref(n);

  thrust_merge(&a, &b, &ref);

  assert(c == ref);

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


