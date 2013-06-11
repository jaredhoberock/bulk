#pragma once

#include <bulk/detail/config.hpp>
#include <thrust/tuple.h>

BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


template<typename Function>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<> &)
{
  f();
}


template<typename Function, typename Arg1>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1> &args)
{
  f(thrust::get<0>(args));
}


template<typename Function, typename Arg1, typename Arg2>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args),
    thrust::get<4>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args),
    thrust::get<4>(args),
    thrust::get<5>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args),
    thrust::get<4>(args),
    thrust::get<5>(args),
    thrust::get<6>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args),
    thrust::get<4>(args),
    thrust::get<5>(args),
    thrust::get<6>(args),
    thrust::get<7>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args),
    thrust::get<4>(args),
    thrust::get<5>(args),
    thrust::get<6>(args),
    thrust::get<7>(args),
    thrust::get<8>(args));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__host__ __device__
void apply_from_tuple(Function f, const thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9,Arg10> &args)
{
  f(thrust::get<0>(args),
    thrust::get<1>(args),
    thrust::get<2>(args),
    thrust::get<3>(args),
    thrust::get<4>(args),
    thrust::get<5>(args),
    thrust::get<6>(args),
    thrust::get<7>(args),
    thrust::get<8>(args),
    thrust::get<9>(args));
}


} // end detail
} // end bulk
BULK_NS_SUFFIX

