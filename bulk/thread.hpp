#pragma once

namespace bulk
{


struct thread
{
  public:
    typedef unsigned int size_type;

    __device__
    size_type index() const
    {
      return threadIdx.x;
    } // end index()
}; // end thread


}; // end bulk

