#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>
#include <math.h>

#define   ELEM_TYPE                 float
#define   ELEM_RAND_RANGE           (100)
#define   ELEM_INIT_VALUE           (1)

#define   PRINT_LINE(title)         printf("============== %s ==============\n", title);

#define   CL_DEVICE_TYPE            CL_DEVICE_TYPE_GPU
#define   LOCAL_WORK_SIZE_POINTER   NULL

#define   BANDWIDTH_CPU_ENABLE
#define   BANDWIDTH_GPU_ENABLE

//#define   NOT_PRINT_FLAG

// OpenCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "matop.h"

int main(int argc, char * argv[]) {
    struct timeval start, end;
    double duration, gflops;

    ELEM_TYPE
      *a_h,
      *a_from_h,
      *a_from_d;

    int
      heightA,
      widthA,
      len,
      ndim = 3,
      run_num;

    size_t
      data_size,
      global_work_size[3] = {1,1,1};

    char
      program_file[] = "",
      kernel_func[] = "";

    if (argc == 9) {
      /*********************************
      1. argc[1] heightA
      2. argc[2] widthA
      3. argc[3] kernel_file_path
      4. argc[4] kernel_func_name
      5. argc[5] run_num

      6. argc[6] global_work_size[0]
      7. argc[7] global_work_size[1]
      8. argc[8] global_work_size[2]
      *********************************/
      heightA = atoi( argv[1] );
      widthA = atoi( argv[2] );
      strcpy( program_file, argv[3] );
      strcpy( kernel_func, argv[4] );
      run_num = atoi( argv[5] );

      global_work_size[0] = atoi( argv[6] );
      global_work_size[1] = atoi( argv[7] );
      global_work_size[2] = atoi( argv[8] );
    }
    else {
      printf("usage: %s HEGHTA WIDTHA KERNEL_FILE_PATH LOOP_EXECUTION_TIME GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[3]\n", argv[0]);
      exit(-1);
    }

    len = heightA * widthA;
    data_size = len * sizeof( ELEM_TYPE );

    a_h = (ELEM_TYPE *) malloc (data_size);
    a_from_h = (ELEM_TYPE *) malloc (data_size);
    a_from_d = (ELEM_TYPE *) malloc (data_size);

    rand_mat(a_h, len, ELEM_RAND_RANGE);
    init_mat(a_from_h, len, ELEM_INIT_VALUE);
    init_mat(a_from_d, len, ELEM_INIT_VALUE);

    PRINT_LINE("INIT")

#ifndef NOT_PRINT_FLAG
    printf("a_h:\n");
    print_mat(a_h, heightA, widthA);
    printf("a_from_h:\n");
    print_mat(a_from_h, heightA, widthA);
    printf("a_from_d:\n");
    print_mat(a_from_d, heightA, widthA);
#endif

    /* cpu copy */
#ifdef BANDWIDTH_CPU_ENABLE
    printf(">>> %d time %s starting...\n", run_num, "CPU");
    gettimeofday(&start, NULL);
    for (int ridx = 0; ridx < run_num; ridx++) {
        copy_mat(a_h, a_from_h, len);
    }
    gettimeofday(&end, NULL);
    duration = ((double)(end.tv_sec-start.tv_sec)*1000000 + 
            (double)(end.tv_usec-start.tv_usec)) / 1000000 / (double) run_num;
    gflops = 1.0 * heightA * widthA;
    printf("%s %d x %d %2.6lf s %2.6lf MFLOPS\n\n", "CPU", heightA, widthA, duration, gflops);
#endif

    PRINT_LINE("CHECK")
    equal_vec(a_h, a_from_h, len);
    printf("a_from_h:\n");
    print_mat(a_from_h, heightA, widthA);


#ifdef BANDWIDTH_GPU_ENABLE
    cl_mem a_h_buff, a_from_d_buff;
    a_h_buff = a_from_d_buff = NULL;

    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;

    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;

    cl_context context = NULL;
    cl_kernel kernel = NULL;
    cl_program program = NULL;

    cl_command_queue command_queue = NULL;
    cl_event event = NULL;
    cl_int ret;

    /* Load the source code and containing the kernel */
    FILE *fp, *program_handle;
    char *program_buffer;
    size_t program_size;

    program_handle = fopen(program_file, "r");
    if (program_handle == NULL) {
        fprintf(stderr, "failed to load kernel.\n");
        exit(-1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *) malloc (program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // Platform
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        printf("failed to get platform ID.%d\n", (int)ret);
        goto error;
    }

    // Device
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE, 1, &device_id, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to get device ID.%d\n", (int)ret);
        goto error;
    }

    // Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create OpenCL context.%d\n", (int)ret);
        goto error;
    }

    // Command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create command queue.%d\n", (int)ret);
        goto error;
    }

    // Memory buffer
    a_h_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
    a_from_d_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, a_h_buff, CL_TRUE, 0, data_size, (void *)a_h, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, a_h_buff, CL_TRUE, 0, data_size, (void *)a_from_d, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to copy data from host to device.\n");
        goto error;
    }

    // Create kernel program from source
    program = clCreateProgramWithSource(context, 1, (const char **)&program_buffer,
          (const size_t *)&program_size, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create OpenCL program from source.%d\n", (int)ret);
        goto error;
    }

    // Build kernel program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to build program.%d\n", (int)ret);
        char build_log[16348];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        printf("Error in kernel: %s\n", build_log);
        goto error;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, kernel_func, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create kernel.%d\n", (int)ret);
        goto error;
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &heightA);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void *) &widthA);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &a_h_buff);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &a_from_d_buff);
    if (ret != CL_SUCCESS) {
        printf("failed to set kernel arguments.%d\n", (int)ret);
        goto error;
    }

    printf(">>> global_work_size[%d]: { %d, %d, %d}\n", ndim, (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2]);
    int global_size = (int) global_work_size[0] * (int) global_work_size[1] * (int) global_work_size[2];
    int task_size = heightA * widthA;
    if (global_size < task_size) {
        printf("[WARN] global work size (%d) is smaller than task size (%d).\n", global_size, task_size);
    }

    /* gpu copy */
    printf(">>> %d times %s starting...\n", run_num, program_file);
    gettimeofday(&start, NULL);
    for (int ridx = 0; ridx < run_num; ridx++) {
        // Run kernel
        clEnqueueNDRangeKernel(command_queue, kernel, ndim, NULL, global_work_size,
                               LOCAL_WORK_SIZE_POINTER, 0, NULL, &event);
        clFinish(command_queue);
    }
    gettimeofday(&end, NULL);
    duration = ((double)(end.tv_sec-start.tv_sec)*1000000 + 
                (double)(end.tv_usec-start.tv_usec)) / 1000000 / (double) run_num;
    gflops = 1.0 * heightA * widthA;
    gflops = gflops / duration * 1.0e-6;
    printf("%s %d x %d %2.6lf s %2.6lf MFLOPS %s\n\n", "GPU", heightA, widthA, duration, gflops, program_file);

    // Copy the output result from device memory
    ret = clEnqueueReadBuffer(command_queue, a_h_buff, CL_TRUE, 0, data_size, (void *)a_from_d, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to copy data from device to host.%d\n", (int)ret);
        goto error;
    }
   
error:
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseMemObject(a_h_buff);
    clReleaseMemObject(a_from_d_buff);
    
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    free(program_buffer);

#endif
    free(a_h);
    free(a_from_h);
    free(a_from_d);

    return 1;
}

