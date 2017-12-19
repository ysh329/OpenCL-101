#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>
#include <math.h>

// ocl
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #define CL_USE_DEPRECATED_OPENCL_1_2_APIS
    #include <CL/cl.h>
#endif

#include "../common/CLutil.h"

/*===================  CHANGE TYPE HERE ===================*/
// Type on Host (CPU)
#define     ELEM_TYPE                       float//__fp16
#define     ELEM_TYPE_STR                   "float"//"__fp16"
// Type on Device (GPU)
#define     CL_ELEM_TYPE                    cl_float//cl_half
#define     CL_ELEM_TYPE_STR                "float"//"half8"

/*=================== OTHER OCL PARAMETERS ================*/
#define     CL_OTHER_MACRO                  ""//" -cl-mad-enable"
#define     OCL_DEVICE_TYPE                 "CL_GPU" // "CL_CPU" or "CL_GPU"
#define     OCL_GLOBAL_WORK_SIZE_DIM        (3)
#define     OCL_LOCAL_WORK_SIZE_DIM         (3)
#define     OCL_LOCAL_WORK_SIZE_POINTER     NULL
#define     KERNEL_FILE_AND_FUNC_MAX_LEN    (200)
#define     OCL_BUILD_LOG_MAX_LEN           (16348)

/*=================== INITIALIZATION ======================*/
#define     ELEM_RAND_RANGE                 (10)
#define     ELEM_INIT_VALUE                 (0)

/*=================== MACRO FUNCTION ======================*/
#define     PRINT_LINE(title)               printf("============== %s ==============\n", title)

/*=================== EXECUTION MODE ======================*/
#define     MATRIX_MULT_CPU_ENABLE
#define     MATRIX_MULT_GPU_ENABLE
#define     MATRIX_ADD_CPU_ENABLE
#define     MATRIX_ADD_GPU_ENABLE
//#define     DONT_PRINT_EACH_BENCHMARK
#define     DONT_PRINT_MATRIX_FLAG
#define     BENCHMARK_SKIP_TIMES            (1)

#include "../common/matop.h"



int main(int argc, char *argv[]) {
    struct timeval start, end;
    double ave_duration, sum_duration, duration, gflops, gbps;
    ELEM_TYPE *a = NULL,
              *b = NULL,
              *c_h = NULL,
              *c_d = NULL;
    char program_file[KERNEL_FILE_AND_FUNC_MAX_LEN],
         kernel_func[KERNEL_FILE_AND_FUNC_MAX_LEN],
         cl_build_program_options[KERNEL_FILE_AND_FUNC_MAX_LEN] = "";
    int m = 0,
        n = 0,
        k = 0,
        len_a = 0,
        len_b = 0,
        len_c = 0,
        cpu_run_num = 0,
        gpu_run_num = 0;
    size_t
        data_size_a = 0,
        data_size_b = 0,
        data_size_c = 0,
        global_work_size[3] = {1,1,1},
        local_work_group_size[3] = {1,1,1};
    char local_work_group_size_str[3];
    if (argc == 14) {
    /*********************************
          0. argc[0] file name
          1. argc[1] m
          2. argc[2] n
          3. argc[3] k
          4. argc[4] kernel_file_path
          5. argc[5] kernel_func_name
          6. argc[6] cpu_run_num
          7. argc[7] gpu_rum_num
          8. argc[8] global_work_size[0]
          9. argc[9] global_work_size[1]
         10. argc[10] global_work_size[2]
         11. argv[11] local_work_group_size[0]
         12. argv[12] local_work_group_size[1]
         13. argv[13] local_work_group_size[2]
    *********************************/
        m                   = atoi( argv[1] );
        n                   = atoi( argv[2] );
        k                   = atoi( argv[3] );
        strcpy(program_file,        argv[4] );
        strcpy(kernel_func,         argv[5] );
        cpu_run_num         = atoi( argv[6] );
        gpu_run_num         = atoi( argv[7] );
        global_work_size[0] = atoi( argv[8] );  // M: global_work_size[1]
        global_work_size[1] = atoi( argv[9] );  // N: global_work_size[0]
        global_work_size[2] = atoi( argv[10] ); // global_work_size[2]:default: 1
        local_work_group_size[0]  = atoi( argv[11] );    // row: the height of work group size
        local_work_group_size[1]  = atoi( argv[12] );    // col: the width of work group size
        local_work_group_size[2]  = atoi( argv[13] );    // depth
    } 
    else {
        printf(">>> [USAGE] %s M N K \\\n", argv[0]);
        printf("            KERNEL_FILE_PATH KERNEL_FUNC_NAME CPU_BENCHMARK_TIMES GPU_BENCHMARK_TIMES \\\n");
        printf("            GLOBAL_WORK_SIZE[0]      GLOBAL_WORK_SIZE[1]      GLOBAL_WORK_SIZE[2] \\\n");
        printf("            LOCAL_WORK_GROUP_SIZE[0] LOCAL_WORK_GROUP_SIZE[1] LOCAL_WORK_GROUP_SIZE[2]\n");
        printf(">>> [ERROR] please input args\n");
        exit(-1);
    }
    // check args
    //for (int i = 0; i < 10; i++) printf("%d %s\n", i, argv[i]);

    // Define work group size
    strcat(cl_build_program_options, " -D WORK_GROUP_ROW=");
    strcat(cl_build_program_options, argv[12]);
    strcat(cl_build_program_options, " -D WORK_GROUP_COL=");
    strcat(cl_build_program_options, argv[11]);
    strcat(cl_build_program_options, " -D WORK_GROUP_DEPTH=");
    strcat(cl_build_program_options, argv[13]);


    strcat(cl_build_program_options, " -D CL_ELEM_TYPE=");
    strcat(cl_build_program_options, CL_ELEM_TYPE_STR);
    strcat(cl_build_program_options, CL_OTHER_MACRO);
    if (strstr(ELEM_TYPE_STR, "short")!=NULL) {
        strcat(cl_build_program_options, " -D CL_INPUT_TYPE=short");
    }
    else if (strstr(ELEM_TYPE_STR, "int")!=NULL) {
        strcat(cl_build_program_options, " -D CL_INPUT_TYPE=int");
    }
    else if (strstr(ELEM_TYPE_STR, "float")!=NULL) {
        strcat(cl_build_program_options, " -D CL_INPUT_TYPE=float");
    }
    else if (strstr(ELEM_TYPE_STR, "double")!=NULL) {
        strcat(cl_build_program_options, " -D CL_INPUT_TYPE=double");
    }
    else if (strstr(ELEM_TYPE_STR, "fp16")!=NULL) {
        strcat(cl_build_program_options, " -D CL_INPUT_TYPE=half");
    }
    else {
        printf(">>> [ERROR] CL_INPUT_TYPE and ELEM_TYPE_STR defination is wrong\n");
        exit(-1);
    }

    printf(">>> [INFO] cl_build_program_options: %s\n", cl_build_program_options);

    PRINT_LINE("INIT");
    printf(">>> [INFO] ELEM_TYPE_STR: %s, sizeof(ELEM_TYPE): %d\n", ELEM_TYPE_STR, (int)sizeof(ELEM_TYPE));
    printf(">>> [INFO] CL_ELEM_TYPE_STR: %s, sizeof(CL_ELEM_TYPE): %d\n", CL_ELEM_TYPE_STR, (int)sizeof(CL_ELEM_TYPE));
    if (sizeof(ELEM_TYPE) != sizeof(CL_ELEM_TYPE)) {
        printf(">>> [WARN] ELEM_TYPE(%s) size differs from CL_ELEM_TYPE(%s)\n", ELEM_TYPE_STR, CL_ELEM_TYPE_STR);
    }

    len_a = m * k;
    len_b = n * k;
    len_c = m * n;
    data_size_a = len_a * sizeof( ELEM_TYPE );
    data_size_b = len_b * sizeof( ELEM_TYPE );
    data_size_c = len_c * sizeof( ELEM_TYPE );
    a   = (ELEM_TYPE *) malloc (data_size_a);
    b   = (ELEM_TYPE *) malloc (data_size_b);
    c_h = (ELEM_TYPE *) malloc (data_size_c);
    c_d = (ELEM_TYPE *) malloc (data_size_c);
    
    printf(">>> [INFO] len_a: %d, len_b: %d, len_c: %d\n", len_a, len_b, len_c);
    printf(">>> [INFO] data_size_a: %d, data_size_b: %d, data_size_c: %d\n\n", (int)data_size_a, (int)data_size_b, (int)data_size_c);
    rand_mat(a,   len_a, ELEM_RAND_RANGE);
    rand_mat(b,   len_b, ELEM_RAND_RANGE);
    init_mat(c_h, len_c, ELEM_INIT_VALUE);
    init_mat(c_d, len_c, ELEM_INIT_VALUE);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("a:\n");
    print_mat(a, k, m);
    printf("b:\n");
    print_mat(b, n, k);
    printf("c_h:\n");
    print_mat(c_h, n, m);
    printf("c_d:\n");
    print_mat(c_d, n, m);
#endif

    /* CPU matrix multiplication */
#ifdef MATRIX_MULT_CPU_ENABLE
    PRINT_LINE("CPU MATRIX MULTPLICATION");
    printf(">>> [INFO] %d times %s starting...\n", cpu_run_num, "CPU");
    sum_duration = 0.0;
    for (int ridx; ridx < (cpu_run_num + BENCHMARK_SKIP_TIMES); ridx++) {
        gettimeofday(&start, NULL);
        mult_mat(a, b, c_h, m, n, k);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) + 
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
        printf("%d \t %.6f\n", ridx, duration);
        #endif
        if (ridx < BENCHMARK_SKIP_TIMES) {
            printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;
    }
    ave_duration = sum_duration / (double)cpu_run_num;
    gflops = 2.0 * m * n * k;
    gflops = gflops / ave_duration *1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", "CPU", m, n, k, ave_duration, gflops);

#endif

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_h:\n");
    print_mat(c_h, n, m);
#endif

#ifdef MATRIX_MULT_GPU_ENABLE
    PRINT_LINE("GPU MATRIX MULTIPLICATION");

    cl_mem             a_buffer          = NULL,
                       b_buffer          = NULL,
                       c_buffer          = NULL;

    cl_platform_id     platform_id       = NULL;
    cl_uint            ret_num_platforms = 0;

    cl_device_id       device_id         = NULL;
    cl_uint            ret_num_devices   = 0;

    cl_context         context           = NULL;
    cl_kernel          kernel            = NULL;
    cl_program         program           = NULL;

    cl_command_queue   command_queue     = NULL;
    cl_event           event             = NULL;
    cl_int             ret               = 0;

    FILE               *program_handle   = NULL;
    char               *program_buffer;
    size_t             program_size;

    // gpu info
    FILE *fp; char buffer[KERNEL_FILE_AND_FUNC_MAX_LEN];
    fp = popen("cat /sys/class/misc/mali0/device/gpuinfo", "r");
    char *ret_ = fgets(buffer, sizeof(buffer), fp);
    printf(">>> [INFO] Device name: %s", buffer);
    pclose(fp);

    // Load source
    printf(">>> [INFO] program_file: %s, kernel_func: %s\n", program_file, kernel_func);
    printf(">>> [INFO] cl_build_program_options: %s\n", cl_build_program_options);
    program_handle = fopen(program_file, "r");
    if (program_handle == NULL) {
        fprintf(stderr, ">>> [ERROR] failed to load kernel\n");
        exit(-1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *) malloc (program_size + 1);
    program_buffer[program_size] = '\0';
    size_t num_read = fread(program_buffer, sizeof(char), program_size, program_handle);
    if (num_read == 0)
        printf(">>> [ERROR] failed to read program file\n");
    fclose(program_handle);

    // Platform
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to get platform ID: %d", (int)ret);
        goto error;
    }

    // Device
    if (strcmp(OCL_DEVICE_TYPE, "CL_GPU") == 0) {
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    }
    else if (strcmp(OCL_DEVICE_TYPE, "CL_CPU") == 0) {
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    }
    else {
        printf(">>> [ERROR] OCL_DEVICE_TYPE declare is wrong\n");
        goto error;
    }
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to get device ID: %d\n", (int)ret);
        goto error;
    }

    // Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to create OpenCL context: %d\n", (int)ret);
        goto error;
    }

    // Command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to create command queue: %d", (int)ret);
        goto error;
    }

    // Memory buffer
    a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size_a, NULL, &ret);
    b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size_b, NULL, &ret);
    c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size_c, NULL, &ret);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d:\n");
    print_mat(c_d, n, m);
#endif

    ret  = clEnqueueWriteBuffer(command_queue, a_buffer, CL_TRUE, 0, data_size_a, (void *)a,   0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, b_buffer, CL_TRUE, 0, data_size_b, (void *)b,   0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, c_buffer, CL_TRUE, 0, data_size_c, (void *)c_d, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from host to device: %d\n", (int)ret);
        goto error;
    }

    // Create kernel
    program = clCreateProgramWithSource(context, 1, (const char **)&program_buffer, 
             (const size_t *)&program_size, &ret); 
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to create OpenCL program from source: %d\n", (int)ret);
        goto error;
    }
    
    // Build kernel program
    ret = clBuildProgram(program, 1, &device_id, cl_build_program_options, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, ">>> [ERROR] failed to build program: %d\n", (int)ret);
        char build_log[OCL_BUILD_LOG_MAX_LEN];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        printf(">>> [ERROR] kernel build log: %s\n", build_log);
        goto error;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, kernel_func, &ret);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to create kernel: %d\n", (int)ret);
        goto error;
    }

    ret  = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &m);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void *) &n);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &k);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &a_buffer);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &b_buffer);
    ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &c_buffer);
    /////////////////////////////////////////////////////////////////////
    ELEM_TYPE alpha = 0.5;
    //ret |= clSetKernelArg(kernel, 6, sizeof(ELEM_TYPE), (void *) &alpha);

    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to set kernel arguments.%d\n", (int)ret);
        goto error;
    }

    printf(">>> [INFO] LOCAL_WORK_SIZE[%d]: (%zu, %zu, %zu)\n", 
            OCL_LOCAL_WORK_SIZE_DIM,
            local_work_group_size[0],
            local_work_group_size[1],
            local_work_group_size[2]);
    printf(">>> [INFO] global_work_size[%d]: { %zu, %zu, %zu }\n", 
           OCL_GLOBAL_WORK_SIZE_DIM, 
           global_work_size[0], 
           global_work_size[1], 
           global_work_size[2]);
    int global_size = (int) global_work_size[0] * 
                      (int) global_work_size[1] * 
                      (int) global_work_size[2];
    int task_size = m * n;
    if (global_size < task_size) {
        printf(">>> [WARN] global work size (%d) is smaller than task size (%d)\n", 
               global_size, 
               task_size);
    }

    /* GPU */
    printf(">>> [INFO] %s %d times %s.%s starting ...\n", 
           OCL_DEVICE_TYPE, 
           gpu_run_num, 
           program_file, 
           kernel_func);
    sum_duration = 0.0;
    for (int ridx = 0; ridx < (gpu_run_num+1); ridx++) {
        gettimeofday(&start, NULL);
        // Run kernel
        //*
        clEnqueueNDRangeKernel(command_queue, kernel, OCL_GLOBAL_WORK_SIZE_DIM, NULL, 
                               global_work_size,
                               OCL_LOCAL_WORK_SIZE_POINTER, 0, NULL, &event);//*/
        //clEnqueueNDRangeKernel(command_queue, kernel, OCL_GLOBAL_WORK_SIZE_DIM, NULL, 
        //                       global_work_size,
        //                       local_work_group_size, 0, NULL, &event);
        // clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
        //                       global_work_size,
        //                       local_work_group_size, 0, NULL, &event);

        clFinish(command_queue);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) + 
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
        printf("%d \t %.6f\n", ridx, duration);
        #endif
        if (ridx < BENCHMARK_SKIP_TIMES) {
            printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;
    }
    ave_duration = sum_duration / (double)gpu_run_num;
    // gflops = 2.0 * m * n * k;
    gflops = 2.0 * m * n * k + m * n;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", 
           OCL_DEVICE_TYPE, 
           m, n, k, 
           ave_duration, gflops);

    // Copy result from device to host
    ret = clEnqueueReadBuffer(command_queue, c_buffer, CL_TRUE, 0, data_size_c, (void *)c_d, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from device to host: %d\n", (int)ret);
        goto error;
    }

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d:\n");
    print_mat(c_d, n, m);
    printf("c_h:\n");
    print_mat(c_h, n, m);
#endif

    //dotprod_mat_alpha(c_h, c_h, len_c, alpha);
    equal_vec(c_h, c_d, len_c);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d:\n");
    print_mat(c_d, n, m);
    printf("c_h:\n");
    print_mat(c_h, n, m);
#endif

/*****************************************/
#ifdef MATRIX_ADD_GPU_ENABLE
    PRINT_LINE("GPU MATRIX ADDITION");
    cl_kernel mat_add_kernel;
    cl_int status = 0;
    ELEM_TYPE beta = 1.3;

    char mat_add_kernel_file[KERNEL_FILE_AND_FUNC_MAX_LEN] = "./gemm_vec4.cl",
         mat_add_kernel_func[KERNEL_FILE_AND_FUNC_MAX_LEN] = "mat_add";//"mat_add";
    if(-1 == opencl_create(&context, &command_queue, &program, cl_build_program_options, mat_add_kernel_file)) {
        printf(">>> [ERROR] OpenCL create fail\n");
        exit(-1);
    }

    c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size_c, NULL, &status);

    status = clEnqueueWriteBuffer(command_queue, c_buffer, CL_TRUE, 0, data_size_c, (void *)c_d, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from host to device: %d\n", (int)status);
        goto error;
    }

    mat_add_kernel = clCreateKernel(program, mat_add_kernel_func, &status);
    checkErr(status, "clCreateKernel() for mat_add_kernel");

    status  = clSetKernelArg(mat_add_kernel, 0, sizeof(cl_int), (void *) &m);
    status |= clSetKernelArg(mat_add_kernel, 1, sizeof(cl_int), (void *) &n);
    status |= clSetKernelArg(mat_add_kernel, 2, sizeof(cl_mem), (void *) &c_buffer);
    status |= clSetKernelArg(mat_add_kernel, 3, sizeof(ELEM_TYPE), (void *) &beta);
    checkErr(status, "clSetKernelArg() for mat_add_kernel");

    global_work_size[0] = n/4;
    global_work_size[1] = m;
    printf(">>> [INFO] global_work_size[%d]: { %d, %d, %d }\n", OCL_GLOBAL_WORK_SIZE_DIM, (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2]);
    global_size = (int) global_work_size[0] * (int) global_work_size[1] * (int) global_work_size[2];
    task_size = m * n;
    if (global_size < task_size) {
        printf(">>> [WARN] global work size (%d) is smaller than task size (%d)\n", global_size, task_size);
    }
    printf(">>> [INFO] %s %d times %s.%s starting ...\n", OCL_DEVICE_TYPE, (int)gpu_run_num, mat_add_kernel_file, mat_add_kernel_func);
    sum_duration = 0.0;
    for (int ridx = 0; ridx < (gpu_run_num+1); ridx++) {
        gettimeofday(&start, NULL);
        clEnqueueNDRangeKernel(command_queue, mat_add_kernel, OCL_GLOBAL_WORK_SIZE_DIM, NULL,
                               global_work_size,
                               OCL_LOCAL_WORK_SIZE_POINTER, 0, NULL, &event);
        clFinish(command_queue);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) +
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
            printf("%d \t %.6f\n", ridx, duration);
        #endif
        if (ridx < BENCHMARK_SKIP_TIMES) {
            printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;
    }
    ave_duration = sum_duration / (double)gpu_run_num;
    gflops = 1.0 * m * n;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", OCL_DEVICE_TYPE, (int)m, (int)n, (int)k, ave_duration, gflops);

    status = clEnqueueReadBuffer(command_queue, c_buffer, CL_TRUE, 0, data_size_c, (void *)c_d, 0, NULL, NULL);
    checkErr(status, "clEnqueueReadBuffer() for mat_add_kernel");
             
/*****************************************/
/*****************************************/
#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_h without alpha:\n");
    print_mat(c_h, n, m);
#endif

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d * alpha:\n");
    print_mat(c_d, n, m);
    printf("c_h * alpha:\n");
    print_mat(c_h, n, m);
#endif

    //dotprod_mat_alpha(c_h, c_h, len_c, (ELEM_TYPE)(beta));
    equal_vec(c_h, c_d, len_c);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d * alpha:\n");
    print_mat(c_d, n, m);
    printf("c_h * alpha:\n");
    print_mat(c_h, n, m);
#endif

#endif

error:
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(program_buffer);

#endif

    free(a);
    free(b);
    free(c_h);
    free(c_d);

    return 1;
}
