#pragma once

#include <bulk/detail/config.hpp>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/counting_iterator.h>


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Difference,
         typename Reference>
class join_iterator;


namespace detail
{


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Difference,
         typename Reference>
struct join_iterator_base
{
  typedef typename thrust::detail::remove_reference<Reference>::type value_type;

  typedef typename thrust::iterator_system<RandomAccessIterator1>::type  system1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type  system2;
  typedef typename thrust::detail::minimum_system<system1,system2>::type system;

  typedef thrust::iterator_adaptor<
    join_iterator<RandomAccessIterator1,RandomAccessIterator2,Difference,Reference>,
    thrust::counting_iterator<Difference>,
    value_type,
    system,
    thrust::random_access_traversal_tag,
    Reference,
    Difference
  > type;
}; // end join_iterator_base


} // end detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Difference = typename thrust::iterator_difference<RandomAccessIterator1>::type,
         typename Reference  = typename thrust::iterator_value<RandomAccessIterator1>::type>
class join_iterator
  : public detail::join_iterator_base<RandomAccessIterator1, RandomAccessIterator2, Difference, Reference>::type
{
  private:
    typedef typename detail::join_iterator_base<RandomAccessIterator1, RandomAccessIterator2, Difference, Reference>::type super_t;
    typedef typename super_t::difference_type size_type;

  public:
    inline __host__ __device__
    join_iterator(RandomAccessIterator1 first1, size_type n, RandomAccessIterator2 first2)
      : super_t(thrust::counting_iterator<size_type>(0)),
        m_n1(n),
        m_iter1(first1),
        m_iter2(first2 - m_n1)
    {}


    inline __host__ __device__
    join_iterator(const join_iterator &other)
      : super_t(other),
        m_n1(other.m_n1),
        m_iter1(other.m_iter1),
        m_iter2(other.m_iter2)
    {}


  private:
    friend class thrust::iterator_core_access;


    __host__ __device__
    typename super_t::reference dereference() const
    {
      size_type i = *super_t::base();
      return (i < m_n1) ? m_iter1[i] : m_iter2[i];
    } // end dereference()


    size_type m_n1;
    RandomAccessIterator1 m_iter1;
    RandomAccessIterator2 m_iter2;
}; // end join_iterator


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
__host__ __device__
join_iterator<RandomAccessIterator1,RandomAccessIterator2,Size> make_join_iterator(RandomAccessIterator1 first1, Size n1, RandomAccessIterator2 first2)
{
  return join_iterator<RandomAccessIterator1,RandomAccessIterator2,Size>(first1, n1, first2);
} // end make_join_iterator()

