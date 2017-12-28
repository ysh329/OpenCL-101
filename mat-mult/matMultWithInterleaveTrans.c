#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <math.h>

/*================= CHANGE TYPE HERE ==================*/
// Type on Host (CPU)
#define     ELEM_TYPE                        float
#define     ELEM_TYPE_STR                    "float"
// Type on Device (GPU)
#define     CL_ELEM_TYPE                     cl_float4
#define     CL_ELEM_TYPE_STR                 "float4"

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


/*================= EXECUTION MODE ====================*/
#define     DEBUG
#define     MATRIX_MULT_CPU_ENABLE
#define     MATRIX_TRANS_CPU_ENABLE
#define     MATRIX_TRANS_GPU_ENABLE
#define     MATRIX_INTERLEAVE_GPU_ENABLE
#define     MATRIX_MULT_GPU_ENABLE

#define     BENCHMARK
#define     PRINT_EACH_BENCHMARK
//#define     DONT_PRINT_MATRIX_FLAG
#define     BENCHMARK_SKIP_TIMES             (1)

/*================= USER DEFINITION LIB ===============*/
#include "../common/CLutil.h"
#include "../common/matop.h"



int main(int argc, char *argv[]) {

#ifdef BENCHMARK
    struct timeval start, end;
    double ave_duration, sum_duration, duration, gflops, gbps;
#endif // BENCHMARK

    ELEM_TYPE *a    = NULL,
              *aI_d = NULL,
              *b    = NULL,
              *bT_h = NULL,
              *bT_d = NULL,
              *c_h  = NULL,
              *c_d  = NULL;
         
    size_t m = 0,
           n = 0,
           k = 0,
           len_a = 0,
           len_b = 0,
           len_c = 0,
           data_size_a = 0,
           data_size_b = 0,
           data_size_c = 0,
           cpu_run_num = 0,
           gpu_run_num = 0,
           mat_trans_global_work_size[OCL_GLOBAL_WORK_SIZE_DIM] = {1, 1, 1},
           mat_interleave_global_work_size[OCL_GLOBAL_WORK_SIZE_DIM] = {1, 1, 1},
           global_work_size[OCL_GLOBAL_WORK_SIZE_DIM] = {1, 1, 1};

    char mat_trans_kernel_file[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         mat_trans_kernel_func[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         mat_interleave_kernel_file[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         mat_interleave_kernel_func[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         mat_mult_kernel_file[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         mat_mult_kernel_func[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN],
         cl_build_program_options[OCL_KERNEL_FILE_AND_FUNC_MAX_LEN] = "-D ";
    
    if (argc == 21) {
    /*********************************
     0. argc[0] file name
     1. argc[1] m
     2. argc[2] n
     3. argc[3] k
     4. argc[4] mat_trans_kernel_file
     5. argc[5] mat_trans_kernel_func
     6. argc[6] mat_mult_kernel_file
     7. argc[7] mat_mult_kernel_func
     8. argc[8] cpu_run_num
     9. argc[9] gpu_rum_num
    10. argc[10] global_work_size[0]
    11. argc[11] global_work_size[1]
    12. argc[12] global_work_size[2]
    *********************************/
        m = atoi( argv[1] );
        n = atoi( argv[2] );
        k = atoi( argv[3] );
        // matrix transpose for B
        strcpy(mat_trans_kernel_file, argv[4]);
        strcpy(mat_trans_kernel_func, argv[5]);
        mat_trans_global_work_size[0] = atoi( argv[7] ); // global_work_size[1]
        mat_trans_global_work_size[1] = atoi( argv[6] ); // global_work_size[0]
        mat_trans_global_work_size[2] = atoi( argv[8] ); // default: 1
        // matrix interleave for A
        strcpy(mat_interleave_kernel_file, argv[9]);
        strcpy(mat_interleave_kernel_func, argv[10]);
        mat_interleave_global_work_size[0] = atoi( argv[12] );
        mat_interleave_global_work_size[1] = atoi( argv[11] );
        mat_interleave_global_work_size[2] = atoi( argv[13] );
        // matrix multiplication for A and B
        strcpy(mat_mult_kernel_file, argv[14]);
        strcpy(mat_mult_kernel_func, argv[15]);
        global_work_size[0] = atoi( argv[17] );  // M: global_work_size[1]
        global_work_size[1] = atoi( argv[16] );  // N: global_work_size[0]
        global_work_size[2] = atoi( argv[18] ); // default global_work_size[2]: 1
        // execution times for gpu and cpu
        cpu_run_num = atoi( argv[19] );
        gpu_run_num = atoi( argv[20] );
    }
    else {
        printf(">>> [USAGE] %s M N K \\\n", argv[0]);
        printf("            TRANS_KERNEL_PATH      TRANS_KERNEL_FUNC_NAME       GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2] \\\n");
        printf("            INTERLEAVE_KERNEL_PATH INTERLEAVE__KERNEL_FUNC_NAME GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2] \\\n");
        printf("            MULT_KERNEL_PATH       MULT_KERNEL_FUNC_NAME        GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2] \\\n");
        printf("            CPU_BENCHMARK_TIMES    GPU_BENCHMARK_TIMES\n");
        printf(">>> [ERROR] please input args\n");
        exit(-1);
    }

    strcat(cl_build_program_options, "CL_ELEM_TYPE=");
    strcat(cl_build_program_options, CL_ELEM_TYPE_STR);
    strcat(cl_build_program_options, OCL_OTHER_MACRO);
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

    PRINT_LINE("INIT");
    printf(">>> [INFO] ELEM_TYPE_STR: %s, sizeof(ELEM_TYPE): %d\n", ELEM_TYPE_STR, (int)sizeof(ELEM_TYPE));
    printf(">>> [INFO] CL_ELEM_TYPE_STR: %s, sizeof(CL_ELEM_TYPE): %d\n", CL_ELEM_TYPE_STR, (int)sizeof(CL_ELEM_TYPE));
    if (sizeof(ELEM_TYPE) != sizeof(CL_ELEM_TYPE)) {
        printf(">>> [WARN] ELEM_TYPE(%s) size differs from CL_ELEM_TYPE(%s)\n", ELEM_TYPE_STR, CL_ELEM_TYPE_STR);
    }

    len_a = m * k;
    len_b = n * k;
    len_c = m * n;

    data_size_a = len_a * sizeof(ELEM_TYPE);
    data_size_b = len_b * sizeof(ELEM_TYPE);
    data_size_c = len_c * sizeof(ELEM_TYPE);

    a    = (ELEM_TYPE *) malloc(data_size_a);
    aI_d = (ELEM_TYPE *) malloc(data_size_a);
    b    = (ELEM_TYPE *) malloc(data_size_b);
    bT_h = (ELEM_TYPE *) malloc(data_size_b);
    bT_d = (ELEM_TYPE *) malloc(data_size_b);
    c_h  = (ELEM_TYPE *) malloc(data_size_c);
    c_d  = (ELEM_TYPE *) malloc(data_size_c);

    rand_mat(a,    len_a, ELEM_RAND_RANGE);
    rand_mat(b,    len_b, ELEM_RAND_RANGE);
    init_mat(aI_d, len_a, ELEM_INIT_VALUE);
    init_mat(bT_h, len_b, ELEM_INIT_VALUE);
    init_mat(bT_d, len_b, ELEM_INIT_VALUE);
    init_mat(c_d,  len_c, ELEM_INIT_VALUE);
    init_mat(c_h,  len_c, ELEM_INIT_VALUE);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("a:\n");      print_mat(a,   k, m);
    printf("b:\n");      print_mat(b,   n, k);
    printf("c_h:\n");    print_mat(c_h, n, m);
    printf("c_d:\n");    print_mat(c_d, n, m);
#endif // DONT_PRINT_MATRIX_FLAG

#ifdef MATRIX_MULT_CPU_ENABLE
    PRINT_LINE("CPU MATRIX MULTIPLICATION");
    printf(">>> [INFO] %d times %s starting...\n", (int)cpu_run_num, "CPU");
    #ifdef BENCHMARK
    sum_duration = 0.0;
    #endif // BENCHMARK
    for (int ridx; ridx < (cpu_run_num + BENCHMARK_SKIP_TIMES); ridx++) {
        gettimeofday(&start, NULL);
        mult_mat(a, b, c_h, m, n, k);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) + 
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
        printf("%d \t %.6f\n", ridx, duration);
        #endif // DONT_PRINT_EACH_BENCHMARK
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
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", "CPU", (int)m, (int)n, (int)k, ave_duration, gflops);

    #ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_h:\n");    print_mat(c_h, n, m);
    #endif // DONT_PRINT_MATRIX_FLAG

#endif // MATRIX_MULT_CPU_ENABLE


#ifdef MATRIX_TRANS_CPU_ENABLE
    // Transpose Matrix B
    PRINT_LINE("CPU MATRIX TRANSPOSE");
    #ifndef DONT_PRINT_MATRIX_FLAG
    //printf("b:\n");
    //print_mat(b, n, k);
    printf("bT_h:\n");
    print_mat(bT_h, k, n);
    #endif
    sum_duration = 0.0;
    for (int ridx; ridx < (cpu_run_num + BENCHMARK_SKIP_TIMES); ridx++) {
        gettimeofday(&start, NULL);
        // todo: inplace cpu mat-trans
        // transpose_mat_inplace(b, k, n, bT_h);
        transpose_mat_naive(b, k, n, bT_h);
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
    gflops = 1.0 * k * m;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%d %2.6lf s %2.6lf GFLOPS\n\n", "CPU-MATRIX-TRANS-FOR-B", (int)k, (int)n, ave_duration, gflops);
    #ifndef DONT_PRINT_MATRIX_FLAG
    //printf("bT_h:\n");
    //print_mat(bT_h, k, n);
    #endif
#endif // MATRIX_TRANS_CPU_ENABLE



#ifdef MATRIX_TRANS_GPU_ENABLE
    PRINT_LINE("GPU MATRIX TRANSPOSE");

    int bT_height = n / 4;
    int bT_width  = 4 * k;

    #ifndef DONT_PRINT_MATRIX_FLAG
    printf("bT_d:\n");
    print_mat(bT_d, bT_width, bT_height);
    #endif // DONT_PRINT_MATRIX_FLAG

    cl_int            status;
    cl_context        context;
    cl_command_queue  command_queue;
    cl_program        program;
    cl_kernel         mat_trans_kernel;
    cl_event          event = NULL;

    print_gpu_info("cat /sys/class/misc/mali0/device/gpuinfo");

	if(-1 == opencl_create(&context, &command_queue, &program, cl_build_program_options, mat_trans_kernel_file))
	{
		printf(">>> [ERROR] OpenCL create fail\n");
        exit(-1);
	}

    cl_mem b_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size_b, NULL, &status);
    cl_mem bT_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size_b, NULL, &status);

    status  = clEnqueueWriteBuffer(command_queue, b_buffer,  CL_TRUE, 0, data_size_b, (void *)b,    0, NULL, NULL);
    status |= clEnqueueWriteBuffer(command_queue, bT_buffer, CL_TRUE, 0, data_size_b, (void *)bT_d, 0, NULL, NULL);
    #ifdef DEBUG
    if (status != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from host to device: %d\n", (int)status);
        goto error;
    }
    #endif // DEBUG
    // create kernel and set kernel args
    mat_trans_kernel = clCreateKernel(program, mat_trans_kernel_func, &status);
    #ifdef DEBUG
    checkErr(status, "clCreateKernel() for mat_trans_kernel");
    #endif // DEBUG

    status  = clSetKernelArg(mat_trans_kernel, 0, sizeof(cl_int), (void *) &k);
    status |= clSetKernelArg(mat_trans_kernel, 1, sizeof(cl_int), (void *) &n);
    status |= clSetKernelArg(mat_trans_kernel, 2, sizeof(cl_mem), (void *) &b_buffer);
    status |= clSetKernelArg(mat_trans_kernel, 3, sizeof(cl_mem), (void *) &bT_buffer);
    #ifdef DEBUG
    checkErr(status, "clSetKernelArg() for mat_trans_kernel");
    #endif // DEBUG

    // estimate global_size and task_size
    #ifdef DEBUG
    printf(">>> [INFO] global_work_size[%d]: { %d, %d, %d }\n", 
                OCL_GLOBAL_WORK_SIZE_DIM, 
                (int)mat_trans_global_work_size[0], 
                (int)mat_trans_global_work_size[1], 
                (int)mat_trans_global_work_size[2]);
    int mat_trans_global_size = (int)mat_trans_global_work_size[0] * 
                                (int)mat_trans_global_work_size[1] * 
                                (int)mat_trans_global_work_size[2];
    int mat_trans_task_size = m * n;
    if (mat_trans_global_size < mat_trans_task_size) {
        printf(">>> [WARN] global work size (%d) is smaller than task size (%d)\n", 
                    mat_trans_global_size, 
                    mat_trans_task_size);
    }
    #endif // DEBUG

    // GPU
    #ifdef DEBUG
    printf(">>> [INFO] %s %d times %s.%s starting ...\n", 
                OCL_DEVICE_TYPE, 
                (int)gpu_run_num, 
                mat_trans_kernel_file, 
                mat_trans_kernel_func);
    sum_duration = 0.0;
    for (int ridx = 0; ridx < (gpu_run_num+1); ridx++) {
        gettimeofday(&start, NULL);
    #endif // DEBUG
        // run kernel
        clEnqueueNDRangeKernel(command_queue, 
                               mat_trans_kernel, 
                               OCL_GLOBAL_WORK_SIZE_DIM, 
                               NULL,
                               mat_trans_global_work_size,
                               OCL_LOCAL_WORK_SIZE_POINTER, 
                               0, 
                               NULL, 
                               &event);
        clFinish(command_queue);
        #ifdef DEBUG
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) +
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
            printf("%d \t %.6f\n", ridx, duration);
        #endif // DONT_PRINT_EACH_BENCHMARK
        if (ridx < BENCHMARK_SKIP_TIMES) {
            printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;
    }
    ave_duration = sum_duration / (double)gpu_run_num;
    gflops = 1.0 * n * k;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%d %2.6lf s %2.6lf GFLOPS\n\n", 
                OCL_DEVICE_TYPE, 
                (int)k, 
                (int)n, 
                ave_duration, 
                gflops);
    #endif // DEBUG

    // Copy result from device to host
    status = clEnqueueReadBuffer(command_queue, 
                                 bT_buffer, 
                                 CL_TRUE, 
                                 0, 
                                 data_size_b, 
                                 (void *)bT_d, 
                                 0, 
                                 NULL, 
                                 NULL);
    //checkErr(status, "clEnqueueReadBuffer() for mat_trans_kernel");

    #ifdef DEBUG
    equal_vec(bT_h, bT_d, len_b);
    #endif

    #ifndef DONT_PRINT_MATRIX_FLAG
    printf("bT_d:\n");
    print_mat(bT_d, bT_width, bT_height);
    #endif // DONT_PRINT_MATRIX_FLAG

#endif // MATRIX_TRANS_GPU_ENABLE





#ifdef MATRIX_INTERLEAVE_GPU_ENABLE
    PRINT_LINE("GPU INTERLEAVE");

    int aI_height = m / 4;
    int aI_width  = k * 4;

    #ifndef DONT_PRINT_MATRIX_FLAG
    printf("aI_d:\n");   
    print_mat(aI_d, aI_width, aI_height);
    #endif

	if (-1 == opencl_create(&context, &command_queue, &program, cl_build_program_options, mat_mult_kernel_file))
	{
		printf(">>> [ERROR] OpenCL create fail\n");
        exit(-1);
	}

    // mat_mult
    cl_mem a_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size_a, NULL, &status);
    cl_mem aI_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size_a, NULL, &status);

    status  = clEnqueueWriteBuffer(command_queue, a_buffer,  CL_TRUE, 0, data_size_a, (void *)a,    0, NULL, NULL);
    status |= clEnqueueWriteBuffer(command_queue, aI_buffer, CL_TRUE, 0, data_size_b, (void *)aI_d, 0, NULL, NULL);

    if (status != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from host to device: %d\n", (int)status);
        goto error;
    }

    // create kernel and set kernel args
    cl_kernel mat_interleave_kernel = clCreateKernel(program, mat_interleave_kernel_func, &status);
    checkErr(status, "clCreateKernel() for mat_interleave_kernel");

	status  = clSetKernelArg(mat_interleave_kernel, 0, sizeof(cl_int), (void *) &m);
    status |= clSetKernelArg(mat_interleave_kernel, 1, sizeof(cl_int), (void *) &k);
    status |= clSetKernelArg(mat_interleave_kernel, 2, sizeof(cl_mem), (void *) &a_buffer);
    status |= clSetKernelArg(mat_interleave_kernel, 3, sizeof(cl_mem), (void *) &aI_buffer);
    checkErr(status, "clSetKernelArg() for mat_interleave_kernel");

    // estimate global_size and task_size
    printf(">>> [INFO] global_work_size[%d]: { %d, %d, %d }\n", 
                OCL_GLOBAL_WORK_SIZE_DIM, 
                (int)global_work_size[0], 
                (int)global_work_size[1], 
                (int)global_work_size[2]);
    int global_size = (int) global_work_size[0] * 
                      (int) global_work_size[1] * 
                      (int) global_work_size[2];
    int task_size = m * n;
    if (global_size < task_size) {
        printf(">>> [WARN] global work size (%d) is smaller than task size (%d)\n", global_size, task_size);
    }

    printf(">>> [INFO] %s %d times %s.%s starting ...\n", 
                OCL_DEVICE_TYPE, 
                (int)gpu_run_num, 
                mat_interleave_kernel_file, 
                mat_interleave_kernel_func);
    sum_duration = 0.0;
    for (int ridx = 0; ridx < (gpu_run_num+1); ridx++) {
        gettimeofday(&start, NULL);
        // run kernel
        clEnqueueNDRangeKernel(command_queue, mat_interleave_kernel, OCL_GLOBAL_WORK_SIZE_DIM, NULL,
                               global_work_size,
                               OCL_LOCAL_WORK_SIZE_POINTER, 0, NULL, &event);
        clFinish(command_queue);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) +
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
            printf("%d \t %.6f\n", ridx, duration);
        #endif // DONT_PRINT_EACH_BENCHMARK
        if (ridx < BENCHMARK_SKIP_TIMES) {
            printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;
    }
    ave_duration = sum_duration / (double)gpu_run_num;
    gflops = 1.0 * m * k;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%d %2.6lf s %2.6lf GFLOPS\n\n", OCL_DEVICE_TYPE, (int)m, (int)k, ave_duration, gflops);

    // Copy result from device to host
    status = clEnqueueReadBuffer(command_queue, aI_buffer, CL_TRUE, 0, data_size_a, (void *)aI_d, 0, NULL, NULL);
    checkErr(status, "clEnqueueReadBuffer() for mat_interleave_kernel");

    #ifndef DONT_PRINT_MATRIX_FLAG
    printf("aI_d:\n");   
    print_mat(aI_d, aI_width, aI_height);
    #endif

#endif // MATRIX_INTERLEAVE_GPU_ENABLE




#ifdef MATRIX_MULT_GPU_ENABLE
    
    PRINT_LINE("GPU MATRIX MULTIPLICATION");
    cl_kernel         mat_mult_kernel;
    // get platform, deviceIDs, create Context, create command_queue,
    // load_cl_source, createProgramWithSource, BuildProgram
	if (-1 == opencl_create(&context, &command_queue, &program, cl_build_program_options, mat_mult_kernel_file))
	{
		printf(">>> [ERROR] OpenCL create fail\n");
        exit(-1);
	}

    // mat_mult
    aI_buffer       = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size_a, NULL, &status);
    bT_buffer       = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size_b, NULL, &status);
    cl_mem c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size_c, NULL, &status);

    status  = clEnqueueWriteBuffer(command_queue, aI_buffer, CL_TRUE, 0, data_size_a, (void *)aI_d, 0, NULL, NULL);
    status |= clEnqueueWriteBuffer(command_queue, bT_buffer, CL_TRUE, 0, data_size_b, (void *)bT_d, 0, NULL, NULL);
    status |= clEnqueueWriteBuffer(command_queue, c_buffer,  CL_TRUE, 0, data_size_c, (void *)c_d,  0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from host to device: %d\n", (int)status);
        goto error;
    }

    // create kernel and set kernel args
    mat_mult_kernel = clCreateKernel(program, mat_mult_kernel_func, &status);
    checkErr(status, "clCreateKernel() for mat_mult_kernel");

	status  = clSetKernelArg(mat_mult_kernel, 0, sizeof(cl_int), (void *) &m);
    status |= clSetKernelArg(mat_mult_kernel, 1, sizeof(cl_int), (void *) &n);
    status |= clSetKernelArg(mat_mult_kernel, 2, sizeof(cl_int), (void *) &k);
    status |= clSetKernelArg(mat_mult_kernel, 3, sizeof(cl_mem), (void *) &a_buffer);
    status |= clSetKernelArg(mat_mult_kernel, 4, sizeof(cl_mem), (void *) &bT_buffer);
    status |= clSetKernelArg(mat_mult_kernel, 5, sizeof(cl_mem), (void *) &c_buffer);
    checkErr(status, "clSetKernelArg() for mat_mult_kernel");

    // estimate global_size and task_size
    printf(">>> [INFO] global_work_size[%d]: { %d, %d, %d }\n", OCL_GLOBAL_WORK_SIZE_DIM, (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2]);
    global_size = (int) global_work_size[0] * (int) global_work_size[1] * (int) global_work_size[2];
    task_size = m * n;
    if (global_size < task_size) {
        printf(">>> [WARN] global work size (%d) is smaller than task size (%d)\n", global_size, task_size);
    }

    // GPU
    printf(">>> [INFO] %s %d times %s.%s starting ...\n", OCL_DEVICE_TYPE, (int)gpu_run_num, mat_mult_kernel_file, mat_mult_kernel_func);
    sum_duration = 0.0;
    for (int ridx = 0; ridx < (gpu_run_num+1); ridx++) {
        gettimeofday(&start, NULL);
        // run kernel
        clEnqueueNDRangeKernel(command_queue, mat_mult_kernel, OCL_GLOBAL_WORK_SIZE_DIM, NULL,
                               global_work_size,
                               OCL_LOCAL_WORK_SIZE_POINTER, 0, NULL, &event);
        clFinish(command_queue);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) +
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        #ifndef DONT_PRINT_EACH_BENCHMARK
            printf("%d \t %.6f\n", ridx, duration);
        #endif // DONT_PRINT_EACH_BENCHMARK
        if (ridx < BENCHMARK_SKIP_TIMES) {
            printf(">>> [INFO] skip first %d time(s)\n", BENCHMARK_SKIP_TIMES);
            continue;
        }
        sum_duration += duration;
    }
    ave_duration = sum_duration / (double)gpu_run_num;
    gflops = 2.0 * m * n * k;
    gflops = gflops / ave_duration * 1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", OCL_DEVICE_TYPE, (int)m, (int)n, (int)k, ave_duration, gflops);

    // Copy result from device to host
    status = clEnqueueReadBuffer(command_queue, c_buffer, CL_TRUE, 0, data_size_c, (void *)c_d, 0, NULL, NULL);
    checkErr(status, "clEnqueueReadBuffer() for mat_mult_kernel");

    equal_vec(c_h, c_d, len_c);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d:\n");
    print_mat(c_d, n, m);
#endif // DONT_PRINT_MATRIX_FLAG

error:
	/* clean openCL resources */
    status = clReleaseCommandQueue(command_queue);
    status = clReleaseProgram(program);
    status = clReleaseContext(context);
    status = clReleaseKernel(mat_trans_kernel);
    status = clReleaseKernel(mat_mult_kernel);
    status = clReleaseMemObject(a_buffer);
    status = clReleaseMemObject(b_buffer);
    status = clReleaseMemObject(c_buffer);

#endif // MATRIX_MULT_GPU_ENABLE
}
