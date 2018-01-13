#include <sys/types.h>
#include <stdio.h>
#include <sys/time.h>

/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */

#define M  (1024*1)
#define N  (1024*1)
#define K  (1024*1)

void init_mat(float *mat, int len, float setVal) 
{
    for (int idx = 0; idx < len; idx++)
        mat[idx] = setVal;
}


int main( void )
{
    static cl_float alpha = 1;

    static cl_float A[M*K] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
    };

    static size_t lda = K;        /* i.e. lda = K */

    static cl_float B[K*N] = {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53,
    };
    static size_t ldb = N;        /* i.e. ldb = N */

    static cl_float beta = 0;

    static cl_float C[M*N] = {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
    };
    static size_t ldc = N;        /* i.e. ldc = N */

    init_mat(A, M*K, 10);
    init_mat(B, K*N, 10);
    init_mat(C, M*N, 10);

    static cl_float result[M*N];

    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Setup clBLAS */
    err = clblasSetup( );

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer( ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
                          NULL, &err );
    bufB = clCreateBuffer( ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
                          NULL, &err );
    bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                          NULL, &err );

    err = clEnqueueWriteBuffer( queue, bufA, CL_TRUE, 0,
            M * K * sizeof( *A ), A, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufB, CL_TRUE, 0,
            K * N * sizeof( *B ), B, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufC, CL_TRUE, 0,
            M * N * sizeof( *C ), C, 0, NULL, NULL );

    struct timeval start_time, end_time;
    double ave_duration = 0, sum_duration = 0, duration = 0, gflops, gbps;
    int run_times=200;

    for (int ridx = 0; ridx < run_times+1; ridx++) {
        gettimeofday(&start_time, NULL);
        /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
        err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                                M, N, K,
                                alpha, bufA, 0, lda,
                                bufB, 0, ldb, beta,
                                bufC, 0, ldc,
                                1, &queue, 0, NULL, &event );

        /* Wait for calculations to be finished. */
        err = clWaitForEvents( 1, &event );
        gettimeofday(&end_time, NULL);
        printf("%2d\t%.6f\n", ridx+1, duration);
        if (ridx >= 1) {
            duration = ((double)(end_time.tv_sec-start_time.tv_sec) + 
                        (double)(end_time.tv_usec-start_time.tv_usec)/1000000);
            sum_duration += duration;
        }

    }
    ave_duration = sum_duration / run_times;
    gflops = 2.0 * M * N * K / ave_duration * 1.0e-9;
    printf(">>> [BENCHMARK] %d x %d x %d \t %.6f sec %2.6lf GFLOPS\n", M, N, K, ave_duration, gflops);

    /* Fetch results of calculations from GPU memory. */
    err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,
                                M * N * sizeof(*result),
                                result, 0, NULL, NULL );

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufC );
    clReleaseMemObject( bufB );
    clReleaseMemObject( bufA );

    /* Finalize work with clBLAS */
    clblasTeardown( );

    /* Release OpenCL working objects. */
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return ret;
}

