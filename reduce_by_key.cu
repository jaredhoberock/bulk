#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/detail/range/tail_flags.h>
#include <thrust/iterator/reverse_iterator.h>
#include <bulk/bulk.hpp>
#include "tail_flags.hpp"
#include "time_invocation_cuda.hpp"


template<unsigned int width, typename T>
__device__
T &element_at(T *sdata, unsigned int row, unsigned int column)
{
  return sdata[row * width + column];
}


template<unsigned int width, typename T>
__device__
T *row(T *sdata, unsigned int row)
{
  return sdata + row * width;
}


template<unsigned int CTA_SIZE,
         unsigned int K,
         bool FullBlock,
         typename Context,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction,
         typename FlagIterator,
         typename FlagType,
         typename IndexType,
         typename ValueType>
__device__ __thrust_forceinline__
void reduce_by_key_body(Context context,
                        const unsigned int n,
                        InputIterator1   ikeys,
                        InputIterator2   ivals,
                        OutputIterator1  okeys,
                        OutputIterator2  ovals,
                        BinaryPredicate  binary_pred,
                        BinaryFunction   binary_op,
                        FlagIterator     iflags,
                        FlagType  (&sflag)[CTA_SIZE],
                        ValueType (&sdata)[CTA_SIZE * K],
                        bool&      carry_in,
                        IndexType& carry_index,
                        ValueType& carry_value)
{
  using thrust::system::cuda::detail::reduce_by_key_detail::load_flags;
  namespace block = thrust::system::cuda::detail::block;

  // load flags
  const FlagType flag_bits  = load_flags<CTA_SIZE,K,FullBlock>(context, n, iflags, sflag);
  const FlagType flag_count = __popc(flag_bits); // TODO hide this behind a template
  const FlagType left_flag  = (context.thread_index() == 0) ? 0 : sflag[context.thread_index() - 1];
  const FlagType head_flag  = (context.thread_index() == 0 || flag_bits & ((1 << (K - 1)) - 1) || left_flag & (1 << (K - 1))) ? 1 : 0;
  
  context.barrier();

  // scan flag counts
  sflag[context.thread_index()] = flag_count; context.barrier();

  block::inclusive_scan(context, sflag, thrust::plus<FlagType>());

  const FlagType output_position = (context.thread_index() == 0) ? 0 : sflag[context.thread_index() - 1];
  const FlagType num_outputs     = sflag[CTA_SIZE - 1];

  context.barrier();

  // shuffle keys and write keys out
  if (!thrust::detail::is_discard_iterator<OutputIterator1>::value)
  {
    // XXX this could be improved
    for (unsigned int i = 0; i < num_outputs; i += CTA_SIZE)
    {
      FlagType position = output_position;

      for(unsigned int k = 0; k < K; k++)
      {
        if (flag_bits & (FlagType(1) << k))
        {
          if (i <= position && position < i + CTA_SIZE)
            sflag[position - i] = K * context.thread_index() + k;
          position++;
        }
      }

      context.barrier();

      if (i + context.thread_index() < num_outputs)
      {
        InputIterator1  tmp1 = ikeys + sflag[context.thread_index()];
        OutputIterator1 tmp2 = okeys + (i + context.thread_index());
        *tmp2 = *tmp1; 
      }
      
      context.barrier();
    }
  }

  // load values
  bulk::static_execution_group<CTA_SIZE,K> g;
  bulk::copy_n(g, ivals, n, sdata);


  // transpose into local array
  ValueType ldata[K];
  for(unsigned int k = 0; k < K; k++)
  {
    ldata[k] = sdata[context.thread_index() * K + k]; 
  }

  // carry in (if necessary)
  if (context.thread_index() == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp1 = carry_value;
    ldata[0] = binary_op(tmp1, ldata[0]);
  }

  context.barrier();

  // sum local values
  {
    for(unsigned int k = 1; k < K; k++)
    {
      const unsigned int offset = K * context.thread_index() + k;

      if (FullBlock || offset < n)
      {
        if (!(flag_bits & (FlagType(1) << (k - 1))))
          ldata[k] = binary_op(ldata[k - 1], ldata[k]);
      }
    }
  }

  // second level segmented scan
  {
    // use head flags for segmented scan
    sflag[context.thread_index()] = head_flag;
    element_at<CTA_SIZE>(sdata, K - 1, context.thread_index()) = ldata[K - 1];
    context.barrier();

    if(FullBlock)
    {
      block::inclusive_scan_by_flag(context, sflag, row<CTA_SIZE>(sdata,K-1), binary_op);
    }
    else
    {
      block::inclusive_scan_by_flag_n(context, sflag, row<CTA_SIZE>(sdata,K-1), n, binary_op);
    }
  }

  // update local values
  if (context.thread_index() > 0)
  {
    unsigned int update_bits  = (flag_bits << 1) | (left_flag >> (K - 1));
// TODO remove guard
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    unsigned int update_count = __ffs(update_bits) - 1u; // NB: this might wrap around to UINT_MAX
#else
    unsigned int update_count = 0;
#endif // THRUST_DEVICE_COMPILER_NVCC

    if (!FullBlock && (K + 1) * context.thread_index() > n)
      update_count = thrust::min(n - K * context.thread_index(), update_count);

    ValueType left = element_at<CTA_SIZE>(sdata,K - 1,context.thread_index() - 1);

    for(unsigned int k = 0; k < K; k++)
    {
      if (k < update_count)
        ldata[k] = binary_op(left, ldata[k]);
    }
  }
  
  context.barrier();

  // store carry out
  if (FullBlock)
  {
    if (context.thread_index() == CTA_SIZE - 1)
    {
      carry_value = ldata[K - 1];
      carry_in    = (flag_bits & (FlagType(1) << (K - 1))) ? false : true;
      carry_index = num_outputs;
    }
  }
  else
  {
    if (context.thread_index() == (n - 1) / K)
    {
      for (unsigned int k = 0; k < K; k++)
          if (k == (n - 1) % K)
              carry_value = ldata[k];
      carry_in    = (flag_bits & (FlagType(1) << ((n - 1) % K))) ? false : true;
      carry_index = num_outputs;
    }
  }

  // shuffle values
  {
    FlagType position = output_position;
  
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = K * context.thread_index() + k;
  
      if (FullBlock || offset < n)
      {
        if (flag_bits & (FlagType(1) << k))
        {
          element_at<CTA_SIZE>(sdata, position / CTA_SIZE, position % CTA_SIZE) = ldata[k];
          position++;
        }
      }
    }
  }

  context.barrier();


  // write values out
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = CTA_SIZE * k + context.thread_index();

    if (offset < num_outputs)
    {
      OutputIterator2 tmp = ovals + offset;
      *tmp = element_at<CTA_SIZE>(sdata, k, context.thread_index());
    }
  }

  context.barrier();
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction,
         typename FlagIterator,
         typename IndexIterator,
         typename ValueIterator,
         typename BoolIterator,
         typename Decomposition,
         typename Context>
struct reduce_by_key_closure
{
  InputIterator1   ikeys;
  InputIterator2   ivals;
  OutputIterator1  okeys;
  OutputIterator2  ovals;
  BinaryPredicate  binary_pred;
  BinaryFunction   binary_op;
  FlagIterator     iflags;
  IndexIterator    interval_counts;
  ValueIterator    interval_values;
  BoolIterator     interval_carry;
  Decomposition    decomp;
  Context          context;

  typedef Context context_type;

  __host__ __device__
  reduce_by_key_closure(InputIterator1   ikeys,
                        InputIterator2   ivals,
                        OutputIterator1  okeys,
                        OutputIterator2  ovals,
                        BinaryPredicate  binary_pred,
                        BinaryFunction   binary_op,
                        FlagIterator     iflags,
                        IndexIterator    interval_counts,
                        ValueIterator    interval_values,
                        BoolIterator     interval_carry,
                        Decomposition    decomp,
                        Context          context = Context())
    : ikeys(ikeys), ivals(ivals), okeys(okeys), ovals(ovals), binary_pred(binary_pred), binary_op(binary_op),
      iflags(iflags), interval_counts(interval_counts), interval_values(interval_values), interval_carry(interval_carry),
      decomp(decomp), context(context) {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    using thrust::system::cuda::detail::detail::uninitialized;

    typedef typename thrust::iterator_value<InputIterator1>::type KeyType;
    typedef typename thrust::iterator_value<InputIterator2>::type ValueType;
    typedef typename Decomposition::index_type                    IndexType;
    typedef typename thrust::iterator_value<FlagIterator>::type   FlagType;

    const unsigned int CTA_SIZE = context_type::ThreadsPerBlock::value;

// TODO centralize this mapping (__CUDA_ARCH__ -> smem bytes)
#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024);
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int SMEM_FIXED = CTA_SIZE * sizeof(FlagType) + sizeof(ValueType) + sizeof(IndexType) + sizeof(bool);
    const unsigned int BOUND_1 = (SMEM - SMEM_FIXED) / ((CTA_SIZE + 1) * sizeof(ValueType));
    const unsigned int BOUND_2 = 8 * sizeof(FlagType);
    const unsigned int BOUND_3 = 6;
  
    // TODO replace this with a static_min<BOUND_1,BOUND_2,BOUND_3>::value
    const unsigned int K = (BOUND_1 < BOUND_2) ? (BOUND_1 < BOUND_3 ? BOUND_1 : BOUND_3) : (BOUND_2 < BOUND_3 ? BOUND_2 : BOUND_3);
  
    __shared__ uninitialized<FlagType[CTA_SIZE]>         sflag;
    //__shared__ uninitialized<ValueType[K][CTA_SIZE]> sdata;
    __shared__ uninitialized<ValueType[CTA_SIZE * K]> sdata;
  
    __shared__ uninitialized<ValueType> carry_value; // storage for carry in and carry out
    __shared__ uninitialized<IndexType> carry_index;
    __shared__ uninitialized<bool>      carry_in; 

    typename Decomposition::range_type interval = decomp[context.block_index()];
    //thrust::system::detail::internal::index_range<IndexType> interval = decomp[context.block_index()];
  

    if (context.thread_index() == 0)
    {
      carry_in = false; // act as though the previous segment terminated just before us
  
      if (context.block_index() == 0)
      {
        carry_index = 0;
      }
      else
      {
        interval_counts += (context.block_index() - 1);
        carry_index = *interval_counts;
      }
    }
  
    context.barrier();
  
    IndexType base = interval.begin();
  
    // advance input and output iterators
    ikeys  += base;
    ivals  += base;
    iflags += base;
    okeys  += carry_index;
    ovals  += carry_index;
  
    const unsigned int unit_size = K * CTA_SIZE;
  
    // process full units
    while (base + unit_size <= interval.end())
    {
      const unsigned int n = unit_size;
      reduce_by_key_body<CTA_SIZE,K,true>(context, n, ikeys, ivals, okeys, ovals, binary_pred, binary_op, iflags, sflag.get(), sdata.get(), carry_in.get(), carry_index.get(), carry_value.get());
      base   += unit_size;
      ikeys  += unit_size;
      ivals  += unit_size;
      iflags += unit_size;
      okeys  += carry_index;
      ovals  += carry_index;
    }
  
    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
    {
      const unsigned int n = interval.end() - base;
      reduce_by_key_body<CTA_SIZE,K,false>(context, n, ikeys, ivals, okeys, ovals, binary_pred, binary_op, iflags, sflag.get(), sdata.get(), carry_in.get(), carry_index.get(), carry_value.get());
    }
  
    if (context.thread_index() == 0)
    {
      interval_values += context.block_index();
      interval_carry  += context.block_index();
      *interval_values = carry_value;
      *interval_carry  = carry_in;
    }
  }
}; // end reduce_by_key_closure


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename IndexIterator,
         typename ValueIterator,
         typename BoolIterator,
         typename BinaryPredicate,
         typename BinaryFunction>
void reduce_by_key_each(thrust::cuda::execution_policy<DerivedPolicy> &exec,
                        InputIterator1  keys_first,
                        InputIterator1  keys_last,
                        InputIterator2  values_first,
                        OutputIterator1 keys_result,
                        OutputIterator2 values_result,
                        IndexIterator   interval_counts_first,
                        IndexIterator   interval_counts_last,
                        ValueIterator   interval_values,
                        BoolIterator    interval_carry,
                        BinaryPredicate binary_pred,
                        BinaryFunction  binary_op)
{
  namespace ns = thrust::system::cuda::detail::reduce_by_key_detail;

  typedef ns::DefaultPolicy<InputIterator1,InputIterator2,OutputIterator1,OutputIterator2,BinaryPredicate,BinaryFunction> Policy;
  typedef typename Policy::Decomposition Decomposition;

  typedef typename Policy::IndexType IndexType;
  typedef typename Policy::FlagType  FlagType;

  Policy policy(keys_first, keys_last);
  Decomposition decomp = policy.decomp;

  typedef tail_flags<InputIterator1,BinaryPredicate,FlagType,IndexType> TailFlags;
  typedef typename TailFlags::iterator FlagIterator;
  TailFlags tail_flags(keys_first, keys_last, binary_pred);
  
  // count number of tail flags per interval
  thrust::system::cuda::detail::reduce_intervals(exec, tail_flags.begin(), interval_counts_first, thrust::plus<IndexType>(), decomp);
  
  thrust::inclusive_scan(exec,
                         interval_counts_first, interval_counts_last,
                         interval_counts_first,
                         thrust::plus<IndexType>());
  
  const unsigned int ThreadsPerBlock = Policy::ThreadsPerBlock;
  typedef thrust::system::cuda::detail::detail::statically_blocked_thread_array<ThreadsPerBlock> Context;
  typedef reduce_by_key_closure<InputIterator1,InputIterator2,OutputIterator1,OutputIterator2,BinaryPredicate,BinaryFunction,
                                FlagIterator,IndexIterator,ValueIterator,BoolIterator,Decomposition,Context> Closure;
  Closure closure
    (keys_first,  values_first,
     keys_result, values_result,
     binary_pred, binary_op,
     tail_flags.begin(),
     interval_counts_first,
     interval_values,
     interval_carry,
     decomp);
  thrust::system::cuda::detail::detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);
} // end reduce_by_key_each()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::pair<OutputIterator1,OutputIterator2>
my_reduce_by_key(InputIterator1 keys_first, 
                 InputIterator1 keys_last,
                 InputIterator2 values_first,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 BinaryPredicate pred,
                 BinaryFunction binary_op)
{
  typedef thrust::cuda::tag DerivedPolicy;
  DerivedPolicy exec;
  namespace ns = thrust::system::cuda::detail::reduce_by_key_detail;

  typedef ns::DefaultPolicy<InputIterator1,InputIterator2,OutputIterator1,OutputIterator2,BinaryPredicate,BinaryFunction> Policy;
  
  Policy policy(keys_first, keys_last);
  
  typedef typename Policy::FlagType       FlagType;
  typedef typename Policy::Decomposition  Decomposition;
  typedef typename Policy::IndexType      IndexType;
  typedef typename Policy::KeyType        KeyType;
  typedef typename Policy::ValueType      ValueType;
  
  // temporary arrays
  typedef thrust::detail::temporary_array<IndexType,DerivedPolicy> IndexArray;
  typedef thrust::detail::temporary_array<ValueType,DerivedPolicy> ValueArray;
  typedef thrust::detail::temporary_array<bool,DerivedPolicy>      BoolArray;
  
  Decomposition decomp = policy.decomp;
  
  // input size
  IndexType n = keys_last - keys_first;
  
  if(n == 0)
  {
    return thrust::make_pair(keys_result, values_result);
  } // end if
  
  IndexArray interval_counts(exec, decomp.size());
  ValueArray interval_values(exec, decomp.size());
  BoolArray  interval_carry(exec, decomp.size());

  // run reduce_by_key over each interval
  reduce_by_key_each(exec,
                     keys_first, keys_last,
                     values_first,
                     keys_result,
                     values_result,
                     interval_counts.begin(),
                     interval_counts.end(),
                     interval_values.begin(),
                     interval_carry.begin(),
                     pred,
                     binary_op);
  
  if(decomp.size() > 1)
  {
    ValueArray interval_values2(exec, decomp.size());
    IndexArray interval_counts2(exec, decomp.size());
    BoolArray  interval_carry2(exec, decomp.size());

    // XXX we should try to eliminate this
    IndexArray interval_counts3(exec, decomp.size());
  
    IndexType N2 = decomp.size();

    reduce_by_key_each(exec,
                       thrust::make_zip_iterator(thrust::make_tuple(interval_counts.begin(), interval_carry.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(interval_counts.end(),   interval_carry.end())),
                       interval_values.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(interval_counts2.begin(), interval_carry2.begin())),
                       interval_values2.begin(),
                       interval_counts3.begin(), interval_counts3.end(),
                       thrust::discard_iterator<>(),
                       thrust::discard_iterator<>(),
                       thrust::equal_to<thrust::tuple<IndexType,bool> >(),
                       binary_op);
  
    thrust::transform_if
      (exec,
       interval_values2.begin(), interval_values2.begin() + N2,
       thrust::make_permutation_iterator(values_result, interval_counts2.begin()),
       interval_carry2.begin(),
       thrust::make_permutation_iterator(values_result, interval_counts2.begin()),
       binary_op,
       thrust::identity<bool>());
  } // end if

  // determine output size
  const IndexType N = interval_counts[interval_counts.size() - 1];
  
  return thrust::make_pair(keys_result + N, values_result + N); 
} // end my_reduce_by_key()


template<typename T>
size_t my_reduce_by_key(const thrust::device_vector<T> *keys,
                        const thrust::device_vector<T> *values,
                        thrust::device_vector<T> *keys_result,
                        thrust::device_vector<T> *values_result)
{
  return my_reduce_by_key(keys->begin(), keys->end(),
                          values->begin(),
                          keys_result->begin(),
                          values_result->begin(),
                          thrust::equal_to<T>(),
                          thrust::plus<T>()).first -
         keys_result->begin();
}


template<typename T>
size_t thrust_reduce_by_key(const thrust::device_vector<T> *keys,
                            const thrust::device_vector<T> *values,
                            thrust::device_vector<T> *keys_result,
                            thrust::device_vector<T> *values_result)
{
  return thrust::reduce_by_key(keys->begin(), keys->end(),
                               values->begin(),
                               keys_result->begin(),
                               values_result->begin()).first -
         keys_result->begin();
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

    return x % 10;
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
  thrust::device_vector<T> keys(n), values(n);
  thrust::device_vector<T> keys_result(n), values_result(n);

  random_fill(keys);
  random_fill(values);

  size_t my_size = my_reduce_by_key(&keys, &values, &keys_result, &values_result);
  double my_msecs = time_invocation_cuda(50, my_reduce_by_key<T>, &keys, &values, &keys_result, &values_result);

  thrust_reduce_by_key(&keys, &values, &keys_result, &values_result);
  double thrust_msecs = time_invocation_cuda(50, thrust_reduce_by_key<T>, &keys, &values, &keys_result, &values_result);

  std::cout << "Thrust's time: " << thrust_msecs << " ms" << std::endl;
  std::cout << "My time:       " << my_msecs << " ms" << std::endl;
  std::cout << "Performance relative to Thrust: " << thrust_msecs / my_msecs << std::endl;

  double my_secs = my_msecs / 1000;

  std::cout << double(n) / my_secs << " keys per second" << std::endl;

  double in_bytes = 2 *sizeof(T) * n;
  double out_bytes = 2 * sizeof(T) * my_size;

  double gigabytes = (in_bytes + out_bytes) / (1 << 30);

  std::cout << gigabytes / my_secs << "GB/s" << std::endl;
}


template<typename T>
void validate(size_t n)
{
  thrust::device_vector<T> keys(n), values(n);
  thrust::device_vector<T> keys_result(n), values_result(n);

  random_fill(keys);
  random_fill(values);

  thrust::device_vector<T> keys_ref(n), values_ref(n);
  size_t thrust_size = thrust_reduce_by_key(&keys, &values, &keys_ref, &values_ref);
  keys_ref.resize(thrust_size);
  values_ref.resize(thrust_size);

  size_t my_size = my_reduce_by_key(&keys, &values, &keys_result, &values_result);
  keys_result.resize(my_size);
  values_result.resize(my_size);

  std::cout << "CUDA error: " << cudaGetErrorString(cudaThreadSynchronize()) << std::endl;

  assert(keys_result == keys_ref);
  assert(values_result == values_ref);
}


int main()
{
  size_t n = 12345678;

  validate<int>(n);

  std::cout << "Large input: " << std::endl;
  std::cout << "int: " << std::endl;
  compare<int>(n);
  std::cout << std::endl;

  std::cout << "float: " << std::endl;
  compare<float>(n);
  std::cout << std::endl;

  std::cout << "double: " << std::endl;
  compare<double>(n);
  std::cout << std::endl;

  return 0;
}

