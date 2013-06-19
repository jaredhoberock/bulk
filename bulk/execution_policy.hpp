#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/execution_group.hpp>

BULK_NS_PREFIX
namespace bulk
{


inline execution_group con(size_t num_threads)
{
  return execution_group(num_threads);
}


template<typename ExecutionGroup>
class group_launch_config
{
  public:
    static const size_t use_default = UINT_MAX;


    typedef ExecutionGroup execution_group_type;


    group_launch_config(cudaStream_t stream,
                        execution_group_type group,
                        size_t num_groups,
                        size_t num_smem_bytes_per_group = use_default)
      : m_stream(stream),
        m_example_group(group),
        m_num_groups(num_groups),
        m_num_smem_bytes_per_group(num_smem_bytes_per_group),
        m_num_threads(num_groups * m_example_group.size())
    {}


    group_launch_config(cudaStream_t stream,
                        size_t num_threads)
      : m_stream(stream),
        m_num_groups(use_default),
        m_example_group(use_default),
        m_num_smem_bytes_per_group(use_default),
        m_num_threads(num_threads)
    {}


    cudaStream_t stream() const
    {
      return m_stream;
    }


    size_t num_groups() const
    {
      return m_num_groups;
    }

    
    size_t num_threads_per_group() const
    {
      return m_example_group.size();
    }


    size_t num_smem_bytes_per_group() const
    {
      return m_num_smem_bytes_per_group;
    }


    size_t num_threads() const
    {
      return m_num_threads;
    }


  private:
    cudaStream_t m_stream;
    execution_group_type m_example_group;
    size_t m_num_groups;
    size_t m_num_smem_bytes_per_group;
    size_t m_num_threads;
};


typedef group_launch_config<execution_group> launch_config;


template<typename ExecutionGroup>
  typename enable_if_execution_group<
    ExecutionGroup,
    group_launch_config<ExecutionGroup>
  >::type
    par_async(cudaStream_t s, ExecutionGroup g, size_t num_groups, size_t num_smem_bytes_per_group = launch_config::use_default)
{
  return group_launch_config<ExecutionGroup>(s, g, num_groups, num_smem_bytes_per_group);
}


template<typename ExecutionGroup>
  typename enable_if_execution_group<
    ExecutionGroup,
    group_launch_config<ExecutionGroup>
  >::type
    par(ExecutionGroup g, size_t num_groups, size_t num_smem_bytes_per_group = launch_config::use_default)
{
  return par_async(0, g, num_groups, num_smem_bytes_per_group);
}


inline launch_config par_async(cudaStream_t s, size_t num_groups, size_t group_size, size_t num_smem_bytes_per_group = launch_config::use_default)
{
  return launch_config(s, con(group_size), num_groups, num_smem_bytes_per_group);
}


inline launch_config par(size_t num_groups, size_t group_size, size_t num_smem_bytes_per_group = launch_config::use_default)
{
  return par_async(0, num_groups, group_size, num_smem_bytes_per_group);
}


inline launch_config par_async(cudaStream_t s, size_t num_threads)
{
  return launch_config(s, num_threads);
}


inline launch_config par(size_t num_threads)
{
  return par_async(0, num_threads);
}


} // end bulk
BULK_NS_SUFFIX

