#include <iostream>
#include <cstdio>
#include <cassert>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/logical.h>

struct saxpy
{
  __host__ __device__
  void operator()(bulk::agent<> &g, float a, float *x, float *y)
  {
    int i = g.index();
    y[i] = a * x[i] + y[i];
  }
};

int main()
{
  size_t n = 1 << 24;
  thrust::device_vector<float> x(n, 1);
  thrust::device_vector<float> y(n, 1);

  float a = 13;

  bulk::async(bulk::par(n), saxpy(), bulk::root.this_exec, a, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()));

  assert(thrust::all_of(y.begin(), y.end(), thrust::placeholders::_1 == 14));

  std::cout << "It worked!" << std::endl;

  return 0;
}

