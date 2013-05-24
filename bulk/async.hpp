#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>
#include <bulk/thread_group.hpp>


namespace bulk
{


template<typename ThreadGroup>
class group_launch_config
{
  public:
    static const size_t use_default = UINT_MAX;


    typedef ThreadGroup thread_group_type;


    group_launch_config(thread_group_type group,
                        size_t num_groups,
                        size_t num_smem_bytes_per_group = use_default)
      : m_example_group(group),
        m_num_groups(num_groups),
        m_num_smem_bytes_per_group(num_smem_bytes_per_group),
        m_num_threads(num_groups * m_example_group.size())
    {}


    group_launch_config(size_t num_threads)
      : m_num_groups(use_default),
        m_example_group(use_default),
        m_num_smem_bytes_per_group(use_default),
        m_num_threads(num_threads)
    {}


    template<typename Function>
    void configure(Function f,
                   typename enable_if_static_thread_group<
                     ThreadGroup,
                     Function
                   >::type * = 0);


    template<typename Function>
    void configure(Function f,
                   typename disable_if_static_thread_group<
                     ThreadGroup,
                     Function
                   >::type * = 0);


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
    thread_group_type m_example_group;
    size_t m_num_groups;
    size_t m_num_smem_bytes_per_group;
    size_t m_num_threads;
};


typedef group_launch_config<thread_group> launch_config;


template<typename ThreadGroup>
  typename enable_if_thread_group<
    ThreadGroup,
    group_launch_config<ThreadGroup>
  >::type
    par(ThreadGroup g, size_t num_groups, size_t num_smem_bytes_per_group = launch_config::use_default)
{
  return group_launch_config<ThreadGroup>(g, num_groups, num_smem_bytes_per_group);
}


inline launch_config par(size_t num_groups, size_t group_size, size_t num_smem_bytes_per_group = launch_config::use_default)
{
  return launch_config(thread_group(group_size), num_groups, num_smem_bytes_per_group);
}


inline launch_config par(size_t num_threads)
{
  return launch_config(num_threads);
}


template<typename LaunchConfig, typename Function>
void async(LaunchConfig l, Function f);


template<typename LaunchConfig, typename Function, typename Arg1>
void async(LaunchConfig l, Function f, Arg1 arg1);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2>
void async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3>
void async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
void async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
void async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5);


} // end bulk

#include <bulk/detail/async.inl>

