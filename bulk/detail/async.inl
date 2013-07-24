#include <bulk/detail/config.hpp>
#include <bulk/async.hpp>
#include <bulk/detail/cuda_launcher.hpp>
#include <bulk/detail/closure.hpp>
#include <bulk/detail/throw_on_error.hpp>


BULK_NS_PREFIX
namespace bulk
{
namespace detail
{


template<typename ExecutionGroup, typename Closure>
future<void> async_in_stream(ExecutionGroup g, Closure c, cudaStream_t s, cudaEvent_t before_event)
{
  if(before_event != 0)
  {
    throw_on_error(cudaStreamWaitEvent(s, before_event, 0), "cudaStreamWaitEvent in async_in_stream");
  }

  bulk::detail::cuda_launcher<ExecutionGroup, Closure> launcher;
  launcher.launch(g, c, s);

  return future_core_access::create_in_stream(s);
}


template<typename ExecutionGroup, typename Closure>
future<void> async(ExecutionGroup g, Closure c)
{
  return async_in_stream(g, c, 0, 0);
} // end async()


template<typename ExecutionGroup, typename Closure>
future<void> async(async_launch<ExecutionGroup> launch, Closure c)
{
  return async_in_stream(launch.exec(), c, launch.stream(), launch.before_event());
} // end async()


} // end detail


template<typename ExecutionGroup, typename Function>
future<void> async(ExecutionGroup g, Function f)
{
  return detail::async(g, detail::make_closure(f));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1)
{
  return detail::async(g, detail::make_closure(f,arg1));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  return detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10));
} // end async()


} // end bulk
BULK_NS_SUFFIX

