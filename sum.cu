#include <cstdio>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <cassert>

struct sum
{
  __device__
  void operator()(bulk::concurrent_group<> &g, thrust::device_ptr<int> data, thrust::device_ptr<int> result)
  {
    unsigned int n = g.size();

    // allocate some special memory that the group can use for fast communication
    int *s_data = static_cast<int*>(bulk::malloc(g, n * sizeof(int)));

    // the whole group cooperatively copies the data
    bulk::copy_n(g, data, n, s_data);

    while(n > 1)
    {
      unsigned int half_n = n / 2;

      if(g.this_exec.index() < half_n)
      {
        s_data[g.this_exec.index()] += s_data[n - g.this_exec.index() - 1];
      }

      // the group synchronizes after each update
      g.wait();

      n -= half_n;
    }

    if(g.this_exec.index() == 0)
    {
      *result = s_data[0];
    }

    // wait for agent 0 to store the result
    g.wait();

    // free the memory cooperatively
    bulk::free(g, s_data);
  }
};

int main()
{
  size_t group_size = 512;

  size_t n = group_size;

  // [1, 1, 1, ... 1] - 512 of them
  thrust::device_vector<int> vec(n, 1);

  thrust::device_vector<int> result(1);

  using bulk::con;

  // let the runtime size the heap
  bulk::async(con(group_size), sum(), bulk::root.this_exec, vec.data(), result.data());

  assert(512 == result[0]);

  // size the heap ourself
  size_t heap_size = group_size * sizeof(int);
  bulk::async(con(group_size, heap_size), sum(), bulk::root.this_exec, vec.data(), result.data());

  assert(512 == result[0]);
}

