# CUDA-learning
## Basic knowledge
### GPU vs. CPU

**CPU** need strong versatility to deal with different type of data (e.g., int, float) and lots of jumping, interrupting caused by logical judgement. The CPU core needs to be very powerful, so lots of hardware resources is needed. As a result, the number of core in a CPU
won't be to many, which limits the ability of parallel computing. 

**GPU**, on the contrary, face more regular data structures. The core of GPU is not as powerful as CPU, but the quantity can be large, which is benificial for parallel computing. 

However, only GPU is not able to complete parallel computing. Serial computing part of the program is executed by CPU, and GPU only performs calculation tasks in parallel part.

## CUDA thread & memory model

### Software

1. **Thread:** Basic unit of parallel computing. Each thread has **registers** and **local memory**.

2. **Thread Block:** composed by a number of threads. Threads in the same block can quickly exchange data through **shared memory**. Thread block can be 1/2/3-dimension.

3. **Grid:** composed by a number of thread blocks. Each grid has **global memory**, **constant memory** and **testure memory** for threads in the grid. A kernel function/program is run on a grid.

![avatar](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919123018799-1605248744.png)

### Hardware
1. **SP:** stream processor or CUDA core, is the basic unit of parallel computing on the hardware level. 

2. **SM:** stream multiprocessor, is composed by a number of SP and other hardware resources (e.g., warp, register, **shared memory**). 

3. **GPU:** (graphics processing unit) composed by a number of SM unit.

**software vs. hardware:** 
1. thread - SP
2. thread block - SM
3. grid - GPU (a kernel is run on a grid/GPU)

![avatar](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919123034967-1110899742.png)

**speed**: register > local memory > shared memory > global memory

## CUDA programming
# Terminology
Host <-> CPU

Device <-> GPU

Kernel <-> function that runs on the device

# API
**cudaMalloc:** allocate global memory on device.

**cudaFree:** release memory on device.

**cudaMemcpy**(void *dst, void *src, size_t nbytes,
enum cudaMemcpyKind direction): transfer data between GPU/CPU.

    cudaMemcpyKind:
    1. cudaMemcpyHostToDevice（CPU->GPU）
    2. cudaMemcpyDeviceToHost（GPU->CPU）
    3. cudaMemcpyDeviceToDevice（GPU->GPU)

**kernel function:**

![avatar](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919123125957-1702896390.png)

    execute code:
    KernelFunc<<<blockNumber, threadPerBlock>>>(args);

# Demo
**matrix multiplication**

    nvcc ./src/matrix_multiplication.cu -o matrix_multiplication
    ./matrix_multiplication.out

## References
[cnblog](https://www.cnblogs.com/skyfsm/p/9673960.html)