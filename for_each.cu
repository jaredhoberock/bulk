#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <bulk/bulk.hpp>

struct for_each_kernel
{
  template<typename Iterator, typename Function>
  __host__ __device__
  void operator()(bulk::agent<> &self, Iterator first, Function f)
  {
    f(first[self.index()]);
  }
};

struct print_functor
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d\n", x);
  }
};

int main()
{
  size_t n = 32;

  thrust::device_vector<int> vec(n);
  thrust::sequence(vec.begin(), vec.end());

  bulk::future<void> f = bulk::async(bulk::par(n), for_each_kernel(), bulk::root.this_exec, vec.begin(), print_functor());

  f.wait();

  return 0;
}

