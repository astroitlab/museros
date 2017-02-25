# import pycuda.driver as drv
# drv.init()
# print "%d device(s) found." % drv.Device.count()
# for ordinal in range(drv.Device.count()):
#     dev = drv.Device(ordinal)
# print "Device #%d: %s" % (ordinal, dev.name())
# print " Compute Capability: %d.%d" % dev.compute_capability()
# print " Total Memory: %s KB" % (dev.total_memory()//(1024))

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

find_max='''

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void max_reduce(const float* const d_array, float* d_max,
                                              const size_t elements)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLOAT_MAX;  // 1

    if (gid < elements)
        shared[tid] = d_array[gid];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);  // 2
        __syncthreads();
    }
    // what to do now?
    // option 1: save block result and launch another kernel
    if (tid == 0)
        d_max[blockIdx.x] = shared[tid]; // 3
    // option 2: use atomics
    if (tid == 0)
      atomicMaxf(d_max, shared[0]);
}

'''


