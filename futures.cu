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
  cudaStream_t s1;
  cudaStreamCreate(&s1);

  using bulk::par;
  using bulk::async;

  // we can insert a task into a stream directly
  bulk::future<void> t1 = async(par(s1, 1), task1());

  // or we can make a new task depend on a previous future
  bulk::future<void> t2 = async(par(t1, 1), task2());

  // task3 is independent of both task1 & task2 and executes in this thread
  task3();

  t1.wait();
  t2.wait();

  cudaStreamDestroy(s1);

  return 0;
}

