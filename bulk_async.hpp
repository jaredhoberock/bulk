#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>


namespace bulk_async
{


class launch
{
  public:
    static const size_t use_default = UINT_MAX;

    launch(size_t num_blocks,
           size_t num_threads_per_block,
           size_t num_smem_bytes_per_block = use_default)
      : m_num_blocks(num_blocks),
        m_num_threads_per_block(num_threads_per_block),
        m_num_smem_bytes_per_block(num_smem_bytes_per_block),
        m_num_threads(num_blocks * num_threads_per_block)
    {}


    launch(size_t num_threads)
      : m_num_blocks(use_default),
        m_num_threads_per_block(use_default),
        m_num_smem_bytes_per_block(use_default),
        m_num_threads(num_threads)
    {}


    template<typename Function>
    void configure(Function f);


    size_t num_blocks() const
    {
      return m_num_blocks;
    }

    
    size_t num_threads_per_block() const
    {
      return m_num_threads_per_block;
    }


    size_t num_smem_bytes_per_block() const
    {
      return m_num_threads_per_block;
    }


    size_t num_threads() const
    {
      return m_num_threads;
    }


  private:
    size_t m_num_blocks;
    size_t m_num_threads_per_block;
    size_t m_num_smem_bytes_per_block;
    size_t m_num_threads;
};


template<typename Function>
void bulk_async(launch l, Function f);


template<typename Function, typename Arg1>
void bulk_async(launch l, Function f, Arg1 arg1);


} // end bulk_async

#include "detail/bulk_async.inl"

