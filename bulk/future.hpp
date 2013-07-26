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

        if(m_owns_stream)
        {
          e = cudaStreamDestroy(m_stream);

          if(e)
          {
            std::cerr << "CUDA error after cudaStreamDestroy in future dtor: " << cudaGetErrorString(e) << std::endl;
          } // end if
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
      : m_stream(0), m_event(0), m_owns_stream(false)
    {}

    // simulate a move
    // XXX need to add rval_ref or something
    future(const future &other)
      : m_stream(0), m_event(0), m_owns_stream(false)
    {
      std::swap(m_stream,      const_cast<future&>(other).m_stream);
      std::swap(m_event,       const_cast<future&>(other).m_event);
      std::swap(m_owns_stream, const_cast<future&>(other).m_owns_stream);
    } // end future()

    // simulate a move
    // XXX need to add rval_ref or something
    future &operator=(const future &other)
    {
      std::swap(m_stream,      const_cast<future&>(other).m_stream);
      std::swap(m_event,       const_cast<future&>(other).m_event);
      std::swap(m_owns_stream, const_cast<future&>(other).m_owns_stream);
      return *this;
    } // end operator=()

  private:
    friend class detail::future_core_access;

    explicit future(cudaStream_t s, bool owns_stream)
      : m_stream(s),m_owns_stream(owns_stream)
    {
      detail::future_detail::throw_on_error(cudaEventCreateWithFlags(&m_event, create_flags), "cudaEventCreateWithFlags in future ctor");
      detail::future_detail::throw_on_error(cudaEventRecord(m_event, m_stream), "cudaEventRecord in future ctor");
    } // end future()

    // XXX this combination makes the constructor expensive
    //static const int create_flags = cudaEventDisableTiming | cudaEventBlockingSync;
    static const int create_flags = cudaEventDisableTiming;

    cudaStream_t m_stream;
    cudaEvent_t m_event;
    bool m_owns_stream;
}; // end future<void>


namespace detail
{


struct future_core_access
{
  inline static future<void> create(cudaStream_t s, bool owns_stream)
  {
    return future<void>(s, owns_stream);
  } // end create_in_stream()

  inline static cudaStream_t stream(const future<void> &f)
  {
    return f.m_stream;
  }

  inline static cudaEvent_t event(const future<void> &f)
  {
    return f.m_event;
  } // end event()
}; // end future_core_access


} // end detail


} // end namespace bulk
BULK_NS_SUFFIX

