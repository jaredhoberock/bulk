bulk
==========

We have a lot parallel work to do, but all we have are these puny threads. Let's Bulk up!

Bulk lets you describe a parallel task as a hierarchical grouping of *execution
agents*. Individually, these agents are like tiny, lightweight threads, but
when grouped together pack some serious muscle. 

We can launch parallel groups of agents with `bulk::async`:

`hello_world.cu`:

```
#include <iostream>
#include <cstdio>
#include <bulk/bulk.hpp>

struct hello
{
  __host__ __device__
  void operator()()
  {
    printf("Hello world!\n");
  }

  __host__ __device__
  void operator()(bulk::parallel_group<> &g)
  {
    printf("Hello world from thread %d\n", g.this_exec.index());
  }
};

int main()
{
  // just launch one agent to say hello
  bulk::async(bulk::par(1), hello());

  // launch 32 agents in parallel
  // bulk::root stands in for the root of the agent hierarchy
  // the hello functor uses this to identify each agent within its group
  bulk::async(bulk::par(32), hello(), bulk::root);

  cudaDeviceSynchronize();

  return 0;
}
```

From here it's a trivial exercise to get to SAXPY:

```
#include <iostream>
#include <cassert>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>

struct saxpy
{
  __host__ __device__
  void operator()(bulk::agent<> &self, float a, float *x, float *y)
  {
    int i = self.index();
    y[i] = a * x[i] + y[i];
  }
};

int main()
{
  size_t n = 1 << 24;
  thrust::device_vector<float> x(n, 1);
  thrust::device_vector<float> y(n, 1);

  float a = 13;

  // pass bulk::root.this_exec so the saxpy functor receives
  // the current execution agent directly
  bulk::async(bulk::par(n), saxpy(), bulk::root.this_exec, a, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()));

  assert(y == thrust::device_vector<float>(n, 14));

  std::cout << "Nice SAXPY. Do you work out?" << std::endl;

  return 0;
}
```

Algorithms built with Bulk are fast.

[`reduce`](reduce.cu) Performance
---------------------

![][32b_float_reduce]
![][64b_float_reduce]

[32b_float_reduce]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdGVQazRVcGxIZGt2TjFybFNpR1hJQmc&oid=2&zx=5u68essty3v7
[64b_float_reduce]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdGVQazRVcGxIZGt2TjFybFNpR1hJQmc&oid=3&zx=kx4rsyamnhnj

[`inclusive_scan`](scan.cu) Performance
----------------------------

![][32b_float_scan]
![][64b_float_scan]

[32b_float_scan]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdGR4cXU4ekdPeXFTOTBTUG9NUDh3OWc&oid=2&zx=5ji93q18pi8m
[64b_float_scan]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdGR4cXU4ekdPeXFTOTBTUG9NUDh3OWc&oid=3&zx=ftlaacipyq13

[`merge`](merge.cu) Performance
-------------------

![][32b_float_merge]
![][64b_float_merge]

[32b_float_merge]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdDE4cm9tTXJWS0RsOTYtNklZSWcxdFE&oid=4&zx=l6i8z7pk97nu
[64b_float_merge]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdDE4cm9tTXJWS0RsOTYtNklZSWcxdFE&oid=5&zx=c8b2ujje3wql

[`reduce_by_key`](reduce_by_key.cu) Performance
---------------------------

![][32b_float_reduce_by_key]
![][64b_float_reduce_by_key]

[32b_float_reduce_by_key]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdDlYWDVhTDZiZXJvYUV6TlF5MUpNSXc&oid=2&zx=4vck6bwpyh52
[64b_float_reduce_by_key]: https://docs.google.com/spreadsheet/oimg?key=0Aj9b9uhQ9hZUdDlYWDVhTDZiZXJvYUV6TlF5MUpNSXc&oid=3&zx=t72yxc8mvorj
