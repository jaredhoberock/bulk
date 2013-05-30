#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>
#include <bulk/future.hpp>


namespace bulk
{


template<typename LaunchConfig, typename Function>
future<void> async(LaunchConfig l, Function f);


template<typename LaunchConfig, typename Function, typename Arg1>
future<void> async(LaunchConfig l, Function f, Arg1 arg1);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5);


} // end bulk

#include <bulk/detail/async.inl>

