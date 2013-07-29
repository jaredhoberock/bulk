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

  // launch 32 agents
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

