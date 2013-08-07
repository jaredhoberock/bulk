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
    printf("Hello world from agent %d\n", g.this_exec.index());
  }
};

int main()
{
  bulk::async(bulk::par(1), hello());

  // wait for this async to finish before exiting the program
  bulk::async(bulk::par(32), hello(), bulk::root).wait();

  return 0;
}

