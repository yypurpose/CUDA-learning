#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <math.h>
#define A_Row  3
#define A_Col 2
#define B_Row  2
#define B_Col 4

__global__ void matrix_mul_gpu(int *A_gpu, int *B_gpu, int *C_gpu, int K, int COL)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    int sum = 0;
    for(int k = 0;k<K;k++)
    {
        sum += A_gpu[row * K + k] * B_gpu[(k * K)+col];
    }
    C_gpu[row * COL + col] = sum;
}

int main()
{
    if (A_Col != B_Row)
        exit(0);

    struct timeval start, end;
    gettimeofday( &start, NULL );

    // malloc host memory
    int *A = (int*)malloc(sizeof(int) * A_Row * A_Col);
    int *B = (int*)malloc(sizeof(int) * B_Row * B_Col);
    int *C = (int*)malloc(sizeof(int) * A_Row * B_Col);

    for (int i = 0; i < A_Row * A_Col; i++) {
        A[i] = 90;
    }
    for (int i = 0; i < B_Row * B_Col; i++) {
        B[i] = 10;
    }

    int *A_gpu, *B_gpu, *C_gpu;

    // malloc device memory
    cudaMalloc((void **)&A_gpu, sizeof(int) * A_Row * A_Col);
    cudaMalloc((void **)&B_gpu, sizeof(int) * B_Row * B_Col);
    cudaMalloc((void **)&C_gpu, sizeof(int) * A_Row * B_Col);

    // copy data from host (CPU) to device (GPU)
    cudaMemcpy(A_gpu, A, sizeof(int) * A_Row * A_Col, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(int) * B_Row * B_Col, cudaMemcpyHostToDevice);

    // The threadPerBlock (x, y) should be a factor of (C_Row, C_Rol) (i.e., (A_Row, B_Col))
    dim3 threadPerBlock(3, 4);
    dim3 blockNumber((A_Row+ threadPerBlock.x -1) / threadPerBlock.x, (B_Col+threadPerBlock.y-1)/ threadPerBlock.y);
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);

    matrix_mul_gpu<<<blockNumber, threadPerBlock>>>(A_gpu, B_gpu, C_gpu, A_Col, B_Col);

    cudaMemcpy(C, C_gpu, sizeof(int) * A_Row * B_Col, cudaMemcpyDeviceToHost);

    for (int i = 0;i<A_Row;i++)
    {
        for (int j = 0;j<B_Col;j++)
        {
            printf("%4d ", C[i * B_Col + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);
}