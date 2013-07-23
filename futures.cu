#include <bulk/bulk.hpp>
#include <cstdio>
#include <typeinfo>

struct task1
{
  __device__
  void operator()()
  {
    printf("Hello world from task1\n");
  }
};

struct task2
{
  __device__
  void operator()()
  {
    printf("Hello world from task2\n");
  }
};

void task3()
{
  printf("Hello world from task3\n");
};

int main()
{
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  using bulk::par;
  using bulk::async;

  bulk::future<void> t1 = async(par(s1, 1), task1());
  bulk::future<void> t2 = async(par(s2, 1), task2());

  task3();

  t1.wait();
  t2.wait();

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  return 0;
}

