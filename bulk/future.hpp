#pragma once

#include <bulk/detail/config.hpp>
#include <utility>
#include <stdexcept>
#include <iostream>

BULK_NS_PREFIX
namespace bulk
{


typedef std::runtime_error future_error;


namespace detail
{


struct future_core_access;


namespace future_detail
{


inline void throw_on_error(cudaError_t e, const char *message)
{
  if(e != cudaSuccess)
  {
    throw future_error(message);
  } // end if
} // end throw_on_error()


} // end future_detail
} // end detail


template<typename T> class future;


template<>
class future<void>
{
  public:
    ~future()
    {
      if(valid())
      {
        // swallow errors
        cudaError_t e = cudaEventDestroy(m_event);

        if(e)
        {
          std::cerr << "CUDA error after cudaEventDestroy in future dtor: " << cudaGetErrorString(e) << std::endl;
        } // end if
      } // end if
    } // end ~future()

    void wait() const
    {
      detail::future_detail::throw_on_error(cudaEventSynchronize(m_event), "cudaEventSynchronize in future::wait");
    } // end wait()

    bool valid() const
    {
      return m_event != 0;
    } // end valid()

    future()
      : m_event(0)
    {}

    // simulate a move
    future(const future &other)
      : m_event(0)
    {
      std::swap(m_event, const_cast<future&>(other).m_event);
    } // end future()

    // simulate a move
    future &operator=(const future &other)
    {
      std::swap(m_event, const_cast<future&>(other).m_event);
      return *this;
    } // end operator=()

  private:
    friend class detail::future_core_access;

    explicit future(cudaStream_t s)
      : m_event(0)
    {
      detail::future_detail::throw_on_error(cudaEventCreateWithFlags(&m_event, create_flags), "cudaEventCreateWithFlags in future ctor");
      detail::future_detail::throw_on_error(cudaEventRecord(m_event, s), "cudaEventRecord in future ctor");
    } // end future()

    // XXX this combination makes the constructor expensive
    //static const int create_flags = cudaEventDisableTiming | cudaEventBlockingSync;
    static const int create_flags = cudaEventDisableTiming;

    cudaEvent_t m_event;
}; // end future<void>


namespace detail
{


struct future_core_access
{
  inline static future<void> create_in_stream(cudaStream_t s)
  {
    return future<void>(s);
  } // end create_in_stream()

  inline static cudaEvent_t event(const future<void> &f)
  {
    return f.m_event;
  } // end event()
}; // end future_core_access


} // end detail


} // end namespace bulk
BULK_NS_SUFFIX

