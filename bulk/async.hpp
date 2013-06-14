#pragma once

#include <bulk/detail/config.hpp>
#include <bulk/future.hpp>
#include <thrust/detail/config.h>
#include <thrust/detail/cstdint.h>


BULK_NS_PREFIX
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


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8);


template<typename LaunchConfig, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
future<void> async(LaunchConfig l, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9);


} // end bulk
BULK_NS_SUFFIX

#include <bulk/detail/async.inl>

