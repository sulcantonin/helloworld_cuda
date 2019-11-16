import time

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda.driver import np_to_array
import  numpy as np
import sys
import os.path
import matplotlib.pyplot as plt


def ceildiv(numer, denom):
    """ceil(numer / denom) for integers."""
    return (numer + denom - 1) / denom


def nearestpow2(num):
    """Returns the closest power of two less than num."""
    return int(2 ** np.floor(np.log2(num)))

cuda.init()
print "initializing GPU resources..",
sys.stdout.flush()

sourcepath = os.path.join(os.path.dirname(__file__), "cudacode.cu")
cudasource = open(sourcepath, "r")
mod = SourceModule(cudasource.read())
cudasource.close()

# finding out stuff about GPI
mrpbattr = cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
mrpb = pycuda.autoinit.device.get_attribute(mrpbattr)

maxblockattr = cuda.device_attribute.MAX_BLOCK_DIM_X
maxblock = pycuda.autoinit.device.get_attribute(maxblockattr)

func = mod.get_function("norm2")
func.prepare("PPi")
func_blocksize = min(maxblock, nearestpow2(mrpb / func.num_regs))


t_gpu = np.zeros(100)
t_cpu = np.zeros(100)

for d in range(1,100):
    data = np.random.rand(500,500,d).astype(np.float32)
    t = time.time()
    expected_output = np.sqrt(np.sum(data ** 2,-1))
    t_cpu[d] = time.time() - t
    print('numpy time {0}'.format(t_cpu[d]))
    dim = data.shape[2]
    data1d = data.reshape((-1,dim))
    output1d = np.zeros((data1d.shape[0]), dtype=np.float32)

    gridcount = ceildiv(data1d.shape[0], func_blocksize)
    maxgpusize = func_blocksize * gridcount * 4


    inputgpu = cuda.mem_alloc(maxgpusize * dim) # input has dim dimension
    outputgpu = cuda.mem_alloc(maxgpusize * 1) # the norm is 1d
    cuda.memcpy_htod(inputgpu,data1d)

    func.prepared_call((gridcount, 1),(func_blocksize, 1, 1),outputgpu,inputgpu, dim)
    t = time.time()
    cuda.memcpy_dtoh(output1d, outputgpu)
    t_gpu[d] = time.time() - t
    print('gpu time {0}'.format(t_gpu[d]))

    # sanity check
    print('L2 error {0}'.format(np.sum(((output1d.reshape( data.shape[0:2]) - expected_output) ** 2).ravel())))

plt.plot(t_cpu)
plt.plot(t_gpu)
plt.show()