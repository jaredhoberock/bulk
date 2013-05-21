#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>
#include <thrust/system/cuda/detail/execution_policy.h>


namespace bulk_async
{


// XXX these should all just be size_t
typedef thrust::detail::uint32_t grid_size_t;

typedef thrust::detail::uint32_t block_size_t;

typedef size_t smem_size_t;


template<typename DerivedPolicy,
         typename Function>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                grid_size_t num_blocks,
                block_size_t num_threads_per_block,
                smem_size_t num_smem_bytes_per_block,
                Function f);


template<typename DerivedPolicy,
         typename Function>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                size_t num_threads,
                Function f);


template<typename DerivedPolicy,
         typename Function,
         typename Arg1>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                grid_size_t num_blocks,
                block_size_t num_threads_per_block,
                smem_size_t num_smem_bytes_per_block,
                Function f,
                Arg1 arg1);


template<typename DerivedPolicy,
         typename Function,
         typename Arg1>
void bulk_async(const thrust::cuda::execution_policy<DerivedPolicy> &exec,
                size_t num_threads,
                Function f,
                Arg1 arg1);


} // end bulk_async

#include "detail/bulk_async.inl"

