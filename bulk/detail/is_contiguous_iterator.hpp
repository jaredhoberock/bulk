#pragma once

#include <thrust/iterator/detail/is_trivial_iterator.h>

namespace bulk
{
namespace detail
{


template<typename T>
  struct is_contiguous_iterator
    : thrust::detail::is_trivial_iterator<T>
{};


} // end detail
} // end bulk

