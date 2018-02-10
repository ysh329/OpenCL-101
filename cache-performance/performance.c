#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <math.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
/*================= CHANGE TYPE HERE ==================*/

#define         VEC_LEN                          (4)
#define         VEC_LEN_STR                      "4"
#define         FP32

#ifdef FP16
// Type on Host (CPU)
#define     ELEM_TYPE                        __fp16 
#define     ELEM_TYPE_STR                    "__fp16"
// Type on Device (GPU)
#define     CL_ELEM_TYPE                     cl_half
#if VEC_LEN == 4
#define CL_ELEM_TYPE_STR "half4"
#else
#define CL_ELEM_TYPE_STR "half8"
#endif
#else
#define     ELEM_TYPE                        float
#define     ELEM_TYPE_STR                    "float"
#define     CL_ELEM_TYPE                     cl_float
#if VEC_LEN == 4
#define CL_ELEM_TYPE_STR "float4"
#else
#define CL_ELEM_TYPE_STR "float8"
#endif
#endif

#define         USE_LOCAL_WORK_SIZE             

/*================= OTHER OCL PARAMETERS ==============*/
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#define     OCL_OTHER_MACRO                  ""//" -cl-mad-enable"
#define     OCL_DEVICE_TYPE                  "CL_GPU" // "CL_CPU" or "CL_GPU"
#define     OCL_GLOBAL_WORK_SIZE_DIM         (3)
#define     OCL_LOCAL_WORK_SIZE_POINTER      NULL
#define     OCL_KERNEL_FILE_AND_FUNC_MAX_LEN (100)
#define     OCL_BUILD_LOG_MAX_LEN            (16348)

/*================= INITIALIZATION ====================*/
#define     ELEM_RAND_RANGE                  (10)
#define     ELEM_INIT_VALUE                  (0)

/*================= MACRO FUNCTION ====================*/
#define     PRINT_LINE(title)                printf("============= %s ============\n", title)
#define     CONCATE(a,b)                     a##b
#define     TOSTR(i)                         #i

/*================= EXECUTION MODE ====================*/
#define     BENCHMARK
#define     PRINT_EACH_BENCHMARK
#define     DONT_PRINT_MATRIX_FLAG
#define     BENCHMARK_SKIP_TIMES             (1)

/*================= USER DEFINITION LIB ===============*/
#include "../common/CLutil.h"
#include "../common/matop.h"

int main(int argc, char *argv[]) {

#ifdef BENCHMARK
    struct timeval start, end;
    double ave_duration = 0, sum_duration = 0, duration = 0, gflops, gbps;
#endif // BENCHMARK

    ELEM_TYPE *a         = NULL;
    size_t len_a         = 0,
           data_size_a   = 0,
           cpu_run_times = 0,
           gpu_run_times = 0,
           global_work_size[OCL_GLOBAL_WORK_SIZE_DIM] = {1, 1, 1},
           local_work_size[OCL_GLOBAL_WORK_SIZE_DIM]  = {1, 1, 1};

    char kernel_file[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         kernel_func[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         cl_build_program_options[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN] = "-cl-mad-enable";

    if (argc == 12) {
        /*********************************
          0. argc[0] file name
          1. argc[1] len_a

          2. argc[2] kernel_file
          3. argc[3] kernel_func

          4. argc[4] global_work_size[0]
          5. argc[5] global_work_size[1]
          6. argc[6] global_work_size[2]

          7. argc[7]mat_mult_local_work_size[0]
          8. argc[8]mat_mult_local_work_size[1]
          9. argc[9]mat_mult_local_work_size[2]

          10. argc[10] cpu_run_times
          11. argc[11] gpu_rum_num
         *********************************/
        len_a = atoi( argv[1] );

        strcpy(kernel_file, argv[2]);
        strcpy(kernel_func, argv[3]);

        global_work_size[0] = atoi( argv[5] ); // global_work_size[1]
        global_work_size[1] = atoi( argv[4] ); // global_work_size[0]
        global_work_size[2] = atoi( argv[6] ); // default: 1

        local_work_size[0] = atoi( argv[8] );
        local_work_size[1] = atoi( argv[7] );
        local_work_size[2] = atoi( argv[9] );

        // execution times for gpu and cpu
        cpu_run_times = atoi( argv[10] );
        gpu_run_times = atoi( argv[11] );
    }
    else {
        printf(">>> [USAGE] %s A_LEN \\\n", argv[0]);
        printf("            KERNEL_PATH            KERNEL_FUNC_NAME \\\n");
        printf("            GLOBAL_WORK_SIZE[0]    GLOBAL_WORK_SIZE[1]    GLOBAL_WORK_SIZE[2] \\\n");
        printf("            LOCAL_WORK_SIZE[0]     LOCAL_WORK_SIZE[1]     LOCAL_WORK_SIZE[2] \\\n");
        printf("            CPU_BENCHMARK_TIMES    GPU_BENCHMARK_TIMES\n");
        printf(">>> [ERROR] please input args\n");
        exit(-1);
    }

    cl_context       context = NULL;
    cl_event         event = NULL;
    cl_kernel        kernel = NULL;
    cl_command_queue command_queue = NULL;
    cl_int           status = 0;
    cl_program       program = NULL;

    sum_duration = 0.0;

    for (int ridx = 0; ridx < (gpu_run_times+1); ridx++) {

        if(-1 == opencl_create(&context, &command_queue, &program, cl_build_program_options, kernel_file))
        {
            printf(">>> [ERROR] OpenCL create fail\n");
            exit(-1);
        }

        cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size_a, NULL, &status);
        kernel = clCreateKernel(program, kernel_func, &status);
        checkErr(status, "clCreateKernel() for benchmark");

        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_buffer);
        checkErr(status, "clSetKernelArg() for benchmark");


        gettimeofday(&start, NULL);
        // run kernel
        clEnqueueNDRangeKernel(command_queue, 
                kernel, 
                OCL_GLOBAL_WORK_SIZE_DIM, 
                NULL,
                global_work_size,
                local_work_size, 
                0, 
                NULL, 
                &event);
        clFinish(command_queue);
        gettimeofday(&end, NULL);

        duration = ((double)(end.tv_sec-start.tv_sec) +
                (double)(end.tv_usec-start.tv_usec)/1000000);
#ifndef DONT_PRINT_EACH_BENCHMARK
        printf("%d \t %.6f\n", ridx, duration);
#endif // DONT_PRINT_EACH_BENCHMARK
        if (ridx < BENCHMARK_SKIP_TIMES) {
            //printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;

        // Copy result from device to host
        status = clEnqueueReadBuffer(command_queue, 
                a_buffer, 
                CL_TRUE, 
                0, 
                data_size_a, 
                (void *)a, 
                0, 
                NULL, 
                NULL);
#ifndef DONT_PRINT_MATRIX_FLAG
        printf("a:\n");
        print_mat(a, len_a, 1);
#endif // DONT_PRINT_MATRIX_FLAG

error:
        /* clean openCL resources */
        status = clReleaseCommandQueue(command_queue);
        status = clReleaseProgram(program);
        status = clReleaseContext(context);
        status = clReleaseKernel(kernel);
        status = clReleaseMemObject(a_buffer);
    }
    ave_duration = sum_duration / (double)gpu_run_times;
    gflops = 2.0 * 100000 * len_a;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %d %2.6lf sec %2.6lf GFLOPS\n\n", 
            OCL_DEVICE_TYPE, 
            (int)len_a, 
            ave_duration, 
            gflops);

}
