#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>
#include "my_matrix_op.h"
#include "matrix_op.h"
#define ASSERT


#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
    {       \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
        exit(1);									\
    }										\
}

//row major
#define OFFSET_ROW(row, col, ld) ((col) + (row) * (ld))

//col major
#define OFFSET_COL(row, col, ld) ((row) + (col) * (ld))

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void cpu_batch_vector_vector_mul(const float *A, const float *B, float *C, int m, int k)
{
    for (int i = 0; i < m; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < k; j++)
        {
            sum += A[i * k + j] * B[i * k + j];
        }
        C[i] = sum;
    }
}

// void batch_vector_vector_mul(const float *A, const float *B, float *C, int m, int k)
// {

// }
//A:col major B:col major
template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int THREAD_SIZE   
    >
__global__ void my_batch_vector_vector_mul_kernel_NN(float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    

    const int THREAD_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE;
    #ifdef ASSERT
    assert(THREAD_PER_BLOCK!=0);
    #endif

    float reg_a[2][THREAD_SIZE];
    float reg_b[2][THREAD_SIZE];
    float reg_c[THREAD_SIZE] = {0};

    // float ldg_a_reg[BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_PER_BLOCK];
    // float ldg_b_reg[BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_PER_BLOCK];
    #ifdef ASSERT
    assert(BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_PER_BLOCK!=0);
    #endif

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_M];

    A = &A[bid * BLOCK_SIZE_M];
    B = &B[bid * BLOCK_SIZE_M];

    //prefetch a tile 将大循环的一次预取称之为tile，tile尺寸为BLOCK_SIZE_M * BLOCK_SIZE_K
    //THREAD_PER_TILE_ROW为负责预取一行ile的线程数
    const int THREAD_PER_TILE_ROW = BLOCK_SIZE_K;
    #ifdef ASSERT
    assert(THREAD_PER_TILE_ROW!=0);
    #endif

    const int tile_x = tid % THREAD_PER_TILE_ROW ;
    const int tile_y = tid / THREAD_PER_TILE_ROW * 4;

    //每次prefetch的tile的高度
    const int TILE_HEIGHT_PER_PRETCH = THREAD_PER_BLOCK / THREAD_PER_TILE_ROW * 4;
    #ifdef ASSERT
    assert(TILE_HEIGHT_PER_PRETCH!=0);
    #endif

    #pragma unroll
    for(int i = 0; i < BLOCK_SIZE_M; i += TILE_HEIGHT_PER_PRETCH)
    {
        FETCH_FLOAT4(As[0][tile_x][tile_y + i]) = FETCH_FLOAT4(A[OFFSET_COL(tile_y + i, tile_x, M)]);
        FETCH_FLOAT4(Bs[0][tile_x][tile_y + i]) = FETCH_FLOAT4(B[OFFSET_COL(tile_y + i, tile_x, M)]);
    }

    __syncthreads();

    //load A from shared memory
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i+=4)
    {
        FETCH_FLOAT4(reg_a[0][i]) = FETCH_FLOAT4(As[0][0][tid * THREAD_SIZE + i]);
    }
    //load B from shared memory
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i+=4)
    {
        FETCH_FLOAT4(reg_b[0][i]) = FETCH_FLOAT4(Bs[0][0][tid * THREAD_SIZE + i]);
    }

    int write_stage = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE_M; i += TILE_HEIGHT_PER_PRETCH)
        {
            FETCH_FLOAT4(As[write_stage][tile_x][tile_y + i]) = FETCH_FLOAT4(A[OFFSET_COL(tile_y + i, tile_x + tile_idx, M)]);
            FETCH_FLOAT4(Bs[write_stage][tile_x][tile_y + i]) = FETCH_FLOAT4(B[OFFSET_COL(tile_y + i, tile_x + tile_idx, M)]);
        }
        //对于每个线程，计算THREAD_SIZE_Y * THREAD_SIZE_X个元素
        //每次小循环先从shared memory预取A的一小列和B的一小行
        //然后计算。小循环的第一次预取上面已经取过了，所以小循环一共循环BLOCK_SIZE_K -1次，
        //最后要补上一次计算，这样就是BLOCK_SIZE_K次预取和计算
        int load_idx = write_stage ^ 1;
        #pragma unroll
        for(int j = 1; j < BLOCK_SIZE_K; j++)
        {
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE; i+=4)
            {
                FETCH_FLOAT4(reg_a[j%2][i]) = FETCH_FLOAT4(As[load_idx][j][tid * THREAD_SIZE + i]);
            }
            //load B from shared memory
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE; i+=4)
            {
                FETCH_FLOAT4(reg_b[j%2][i]) = FETCH_FLOAT4(Bs[load_idx][j][tid * THREAD_SIZE + i]);
            }
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE; i++)
            {

                reg_c[i] += reg_a[(j-1)%2][i] * reg_b[(j-1)%2][i];
            }
        }
        
        //大迭代还差加载到shared memory
        if(tile_idx < K)
        {
            write_stage ^= 1;
            __syncthreads();
            
        }
        #pragma unroll
        for(int i = 0; i < THREAD_SIZE; i+=4)
        {
            FETCH_FLOAT4(reg_a[BLOCK_SIZE_K%2][i]) = FETCH_FLOAT4(As[load_idx^1][0][tid * THREAD_SIZE + i]);
        }
        //load B from shared memory
        #pragma unroll
        for(int i = 0; i < THREAD_SIZE; i+=4)
        {
            FETCH_FLOAT4(reg_b[BLOCK_SIZE_K%2][i]) = FETCH_FLOAT4(Bs[load_idx^1][0][tid * THREAD_SIZE + i]);
        }
        #pragma unroll
        for(int i = 0; i < THREAD_SIZE; i++)
        {

            reg_c[i] += reg_a[(BLOCK_SIZE_K-1)%2][i] * reg_b[(BLOCK_SIZE_K-1)%2][i];
        }
    }while(tile_idx < K);

    //将计算结果写回到C矩阵
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i+=4)
    {
        FETCH_FLOAT4(C[OFFSET_ROW(
            BLOCK_SIZE_M * bid + tid * THREAD_SIZE + i,
            0,
            1
        )])=FETCH_FLOAT4(reg_c[i]);
    }

}


//A:row major B:row major
template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int THREAD_SIZE   
    >
__global__ void my_batch_vector_vector_mul_kernel(float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const uint8_t c = 108;

    const int THREAD_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE;
    #ifdef ASSERT
    assert(THREAD_PER_BLOCK!=0);
    #endif

    float reg_a[2][THREAD_SIZE];
    float reg_b[2][THREAD_SIZE];
    float reg_c[THREAD_SIZE] = {0};

    float ldg_a_reg[BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_PER_BLOCK];
    float ldg_b_reg[BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_PER_BLOCK];
    #ifdef ASSERT
    assert(BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_PER_BLOCK!=0);
    #endif

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_M];

    A = &A[bid * BLOCK_SIZE_M * K];
    B = &B[bid * BLOCK_SIZE_M * K];

    //prefetch a tile 将大循环的一次预取称之为tile，tile尺寸为BLOCK_SIZE_M * BLOCK_SIZE_K
    //THREAD_PER_TILE_ROW为负责预取一行tile的线程数
    const int THREAD_PER_TILE_ROW = BLOCK_SIZE_K / 4;
    #ifdef ASSERT
    assert(THREAD_PER_TILE_ROW!=0);
    #endif

    const int tile_x = tid % THREAD_PER_TILE_ROW * 4;
    const int tile_y = tid / THREAD_PER_TILE_ROW;

    //每次prefetch的tile的高度
    const int TILE_HEIGHT_PER_PRETCH = THREAD_PER_BLOCK / THREAD_PER_TILE_ROW;
    #ifdef ASSERT
    assert(TILE_HEIGHT_PER_PRETCH!=0);
    #endif

    #pragma unroll
    for(int i = 0; i < BLOCK_SIZE_M; i +=TILE_HEIGHT_PER_PRETCH)
    {
        int ldg_idx = i / TILE_HEIGHT_PER_PRETCH * 4;
        //从global memory 加载到寄存器
        FETCH_FLOAT4(ldg_a_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET_ROW(i + tile_y, tile_x, K)]);
        //再从寄存器加载到shared memory，此时从行主序变成了列主序
        As[0][tile_x + 0][i + tile_y] = ldg_a_reg[ldg_idx + 0];
        As[0][tile_x + 1][i + tile_y] = ldg_a_reg[ldg_idx + 1];
        As[0][tile_x + 2][i + tile_y] = ldg_a_reg[ldg_idx + 2];
        As[0][tile_x + 3][i + tile_y] = ldg_a_reg[ldg_idx + 3];

    }

    #pragma unroll
    for(int i = 0; i < BLOCK_SIZE_M; i +=TILE_HEIGHT_PER_PRETCH)
    {
        int ldg_idx = i / TILE_HEIGHT_PER_PRETCH * 4;
        //从global memory 加载到寄存器
        FETCH_FLOAT4(ldg_b_reg[ldg_idx]) = FETCH_FLOAT4(B[OFFSET_ROW(i + tile_y, tile_x, K)]);
        //再从寄存器加载到shared memory，此时从行主序变成了列主序
        Bs[0][tile_x + 0][i + tile_y] = ldg_b_reg[ldg_idx + 0];
        Bs[0][tile_x + 1][i + tile_y] = ldg_b_reg[ldg_idx + 1];
        Bs[0][tile_x + 2][i + tile_y] = ldg_b_reg[ldg_idx + 2];
        Bs[0][tile_x + 3][i + tile_y] = ldg_b_reg[ldg_idx + 3];
    }

    __syncthreads();

    //load A from shared memory
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i+=4)
    {
        FETCH_FLOAT4(reg_a[0][i]) = FETCH_FLOAT4(As[0][0][tid * THREAD_SIZE + i]);
    }
    //load B from shared memory
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i+=4)
    {
        FETCH_FLOAT4(reg_b[0][i]) = FETCH_FLOAT4(Bs[0][0][tid * THREAD_SIZE + i]);
    }

    int write_stage = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        if(tile_idx < K)
        {
            //大迭代中预取下一次要计算的tile，总共执行K/BLOCK_SIZE_K-1次，因为第一次已经在上面预取过了
            //整个预取分为两个部分，从global memory到寄存器，再从寄存器到shared memory
            //目前这部分是从global memory到寄存器
            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE_M; i +=TILE_HEIGHT_PER_PRETCH)
            {
                int ldg_idx = i / TILE_HEIGHT_PER_PRETCH * 4;
                //从global memory 加载到寄存器
                FETCH_FLOAT4(ldg_a_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET_ROW(i + tile_y, tile_x + tile_idx, K)]);
                
            }

            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE_M; i +=TILE_HEIGHT_PER_PRETCH)
            {
                int ldg_idx = i / TILE_HEIGHT_PER_PRETCH * 4;
                //从global memory 加载到寄存器
                FETCH_FLOAT4(ldg_b_reg[ldg_idx]) = FETCH_FLOAT4(B[OFFSET_ROW(i + tile_y, tile_x + tile_idx, K)]);
                
            }
            
        }
        //对于每个线程，计算THREAD_SIZE_Y * THREAD_SIZE_X个元素
        //每次小循环先从shared memory预取A的一小列和B的一小行
        //然后计算。小循环的第一次预取上面已经取过了，所以小循环一共循环BLOCK_SIZE_K -1次，
        //最后要补上一次计算，这样就是BLOCK_SIZE_K次预取和计算
        int load_idx = write_stage ^ 1;
        #pragma unroll
        for(int j = 1; j < BLOCK_SIZE_K; j++)
        {
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE; i+=4)
            {
                FETCH_FLOAT4(reg_a[j%2][i]) = FETCH_FLOAT4(As[load_idx][j][tid * THREAD_SIZE + i]);
            }
            //load B from shared memory
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE; i+=4)
            {
                FETCH_FLOAT4(reg_b[j%2][i]) = FETCH_FLOAT4(Bs[load_idx][j][tid * THREAD_SIZE + i]);
            }
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE; i++)
            {
                float a = reg_a[(j-1)%2][i];
                // float b = reg_b[(j-1)%2][i];
                reg_c[i] += reg_a[(j-1)%2][i] * reg_b[(j-1)%2][i];
                // if(bid==0&&tid==0&&i==0&&j==1&&tile_idx == BLOCK_SIZE_K)
                // {
                //     printf("a: %f, u_int_32 a: 0x%x, u_int_8 a: 0x%x\n",a,*((uint32_t*)(&a)),*((uint8_t*)(&(reg_a[(j-1)%2][i]))+3));
                //     printf("b: %f, u_int_32 b: 0x%x, u_int_8 b: 0x%x\n",b,*((uint32_t*)(&b)),*((uint8_t*)(&(reg_b[(j-1)%2][i]))+3));
                //     printf("c:%d\n",temp);
                // }
                    

                // reg_c[i] += reg_a[(j-1)%2][i] * reg_b[(j-1)%2][i];
                
            }
        }
        
        //大迭代的预取还差从寄存器到shared memory
        if(tile_idx < K)
        {
            
            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE_M; i +=TILE_HEIGHT_PER_PRETCH)
            {
                int ldg_idx = i / TILE_HEIGHT_PER_PRETCH * 4;
                //从寄存器加载到shared memory，此时从行主序变成了列主序
                As[write_stage][tile_x + 0][i + tile_y] = ldg_a_reg[ldg_idx + 0];
                As[write_stage][tile_x + 1][i + tile_y] = ldg_a_reg[ldg_idx + 1];
                As[write_stage][tile_x + 2][i + tile_y] = ldg_a_reg[ldg_idx + 2];
                As[write_stage][tile_x + 3][i + tile_y] = ldg_a_reg[ldg_idx + 3];
            }

            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE_M; i +=TILE_HEIGHT_PER_PRETCH)
            {
                int ldg_idx = i / TILE_HEIGHT_PER_PRETCH * 4;
                //从寄存器加载到shared memory，此时从行主序变成了列主序
                Bs[write_stage][tile_x + 0][i + tile_y] = ldg_b_reg[ldg_idx + 0];
                Bs[write_stage][tile_x + 1][i + tile_y] = ldg_b_reg[ldg_idx + 1];
                Bs[write_stage][tile_x + 2][i + tile_y] = ldg_b_reg[ldg_idx + 2];
                Bs[write_stage][tile_x + 3][i + tile_y] = ldg_b_reg[ldg_idx + 3];
            }

            write_stage ^= 1;
            __syncthreads();
            
        }
        #pragma unroll
        for(int i = 0; i < THREAD_SIZE; i+=4)
        {
            FETCH_FLOAT4(reg_a[BLOCK_SIZE_K%2][i]) = FETCH_FLOAT4(As[load_idx^1][0][tid * THREAD_SIZE + i]);
        }
        //load B from shared memory
        #pragma unroll
        for(int i = 0; i < THREAD_SIZE; i+=4)
        {
            FETCH_FLOAT4(reg_b[BLOCK_SIZE_K%2][i]) = FETCH_FLOAT4(Bs[load_idx^1][0][tid * THREAD_SIZE + i]);
        }
        #pragma unroll
        for(int i = 0; i < THREAD_SIZE; i++)
        {
            int a = reg_a[(BLOCK_SIZE_K-1)%2][i];
            // uint8_t temp = (*((uint8_t*)(&(reg_a[(BLOCK_SIZE_K-1)%2][i]))+3)+*((uint8_t*)(&(reg_b[(BLOCK_SIZE_K-1)%2][i]))+3));
            reg_c[i] += reg_a[(BLOCK_SIZE_K-1)%2][i] * reg_b[(BLOCK_SIZE_K-1)%2][i];
        }
    }while(tile_idx < K);

    //将计算结果写回到C矩阵
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE; i+=4)
    {
        FETCH_FLOAT4(C[OFFSET_ROW(
            BLOCK_SIZE_M * bid + tid * THREAD_SIZE + i,
            0,
            1
        )])=FETCH_FLOAT4(reg_c[i]);
    }

}

template <
    const int BLOCK_SIZE_M
    >
__global__ void batch_vv_naive(
    float *A,
    float *B,
    float *C,
    const int M,
    const int K)
    {
        int bx = blockIdx.x;
        int tx = threadIdx.x;

        int row = bx * BLOCK_SIZE_M + tx;
        C[row ] = 0.0f;
        for(int i = 0; i < K; i++)
        {
            C[row] += A[row * K + i] * B[row * K + i];
        }

    }
//trans:0 row major, 1 col major
void print_matrix(float *A, int M, int K, bool trans)
{
    if(trans==0)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < K; j++)
            {
                printf("%f ", A[i * K + j]);
            }
            printf("\n");
        }
    }
    else
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < K; j++)
            {
                printf("%f ", A[i + j * M]);
            }
            printf("\n");
        }
    }
}

int main()
{
    cudaSetDevice(0);
    int M=131072;
    int K=256;

    const int BLOCK_SIZE_M = 16;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE = 4;
    assert(BLOCK_SIZE_K >= 4);
    assert(THREAD_SIZE >= 4);

    const int iter = 10000;

    size_t size_A = sizeof(float) * M * K;
    size_t size_B = sizeof(float) * M * K;
    size_t size_C = sizeof(float) * M;

    size_t size_A_d = sizeof(double) * M * K;
    size_t size_B_d = sizeof(double) * M * K;
    size_t size_C_d = sizeof(double) * M;


    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_C);

    double* h_A_d = (double*)malloc(size_A_d);
    double* h_B_d = (double*)malloc(size_B_d);
    double* h_C_d = (double*)malloc(size_C_d);

    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = (float)((float)2/13300000.0);
        h_A_d[i] = (double)(i % 13);
    }

    for (int i = 0; i < M * K; i++)
    {
        h_B[i] = (float)(2.0/84300000.0);
        h_B_d[i] = (double)(i % 17);
    }

    for (int i = 0; i < M; i++)
    {
        h_C[i] = ((float)(2));
        h_C_d[i] = (double)(i % 19);
    }

    float* d_A;
    float* d_B;
    float* d_C;
    double* d_A_d;
    double* d_B_d;
    double* d_C_d;

    checkCudaErrors(cudaMalloc(&d_A, size_A));
    checkCudaErrors(cudaMalloc(&d_B, size_B));
    checkCudaErrors(cudaMalloc(&d_C, size_C));

    checkCudaErrors(cudaMalloc(&d_A_d, size_A_d));
    checkCudaErrors(cudaMalloc(&d_B_d, size_B_d));
    checkCudaErrors(cudaMalloc(&d_C_d, size_C_d));

    checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_A_d, h_A_d, size_A_d, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B_d, h_B_d, size_B_d, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C_d, h_C_d, size_C_d, cudaMemcpyHostToDevice));

    dim3 dimBlock((BLOCK_SIZE_M + THREAD_SIZE - 1) / THREAD_SIZE);
    dim3 dimGrid((M+BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    cudaDeviceSynchronize();
    cudaEvent_t start, stop, start_cublas, stop_cublas;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_cublas);
    cudaEventCreate(&stop_cublas);
    float msecTotal = 0.0f;
    float msecTotal_cublas = 0.0f;
    
    //warm up
    batch_vv_naive<BLOCK_SIZE_M><<<(M+BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, BLOCK_SIZE_M>>>(d_A, d_B, d_C, M, K);

    cudaEventRecord(start);
    for(int i = 0; i < iter; i++)
    {
        my_batch_vector_vector_mul_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K,THREAD_SIZE><<<dimGrid,dimBlock>>>(d_A, d_B, d_C, M, K);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    checkCudaErrors(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    long  totalOp = 2 * M * K;
    double my_gflops = 0.0f, cublas_gflops = 0.0f;
    my_gflops = (double)(totalOp * iter) / msecTotal / 1e6;
    printf("my performance : %.2f GFlops/s, total op: %ld avg_time:%fms\n", my_gflops, totalOp, (double)(msecTotal) / iter);
    //cublas
    cudaEventRecord(start_cublas,0);
    for(int i = 0; i < iter; i++)
    {
        matrix_op::bmm(d_A_d,{M,1,K},d_B_d,{M,K,1},d_C_d);
    }
    
    cudaEventRecord(stop_cublas,0);
    cudaEventSynchronize(stop_cublas);
    cudaEventElapsedTime(&msecTotal_cublas, start_cublas, stop_cublas);
    cublas_gflops = (double)(totalOp * iter) / msecTotal_cublas / 1e6;
    printf("cublas performance : %.2f GFlops/s, total op: %ld avg_time:%fms\n", cublas_gflops, totalOp, (double)(msecTotal_cublas) / iter);

    //verify correctness
    cpu_batch_vector_vector_mul(h_A, h_B, h_D, M, K);
    double eps = 1e-6;
    bool ifprint = false;
    bool correct = true;
    for(int i = 0; i < M ; i++)
    {
        if(abs(h_C[i] - h_D[i]) > eps)
        {
            printf("error! idx:%d\n h_C:%.8f\n h_D:%.8f\n", i, h_C[i], h_D[i]);
            correct = false;
            break;
        }
    }
    if(correct)
    {
        printf("correct!\n");
    }
    else
    {
        printf("wrong!\n");
        
    }
    if(ifprint)
    {
        printf("A:\n");
        print_matrix(h_A, M, K, 1);
        printf("B:\n");
        print_matrix(h_B, M, K, 1);
        printf("C:\n");
        print_matrix(h_C, M, 1, 0);
        printf("D:\n");
        print_matrix(h_D, M, 1, 0);
    }
}