#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>
#include <math.h>

	// ocl
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

	/*===================  CHANGE TYPE HERE ===================*/
	// Type on Host (CPU)
#define     ELEM_TYPE                       float
#define     ELEM_TYPE_STR                   "float"
	// Type on Device (GPU)
#define     CL_ELEM_TYPE                    cl_float
#define     CL_ELEM_TYPE_STR                "float"

	/*=================== OTHER OCL PARAMETERS ================*/
#define     CL_OTHER_MACRO                  " -cl-mad-enable"
#define     OCL_DEVICE_TYPE                 "CL_GPU" // "CL_CPU" or "CL_GPU"
#define     GLOBAL_WORK_SIZE_DIM            (3)
#define     LOCAL_WORK_SIZE_POINTER         NULL
#define     KERNEL_FILE_AND_FUNC_MAX_LEN    (100)
#define     OCL_BUILD_LOG_MAX_LEN           (16348)

	/*=================== INITIALIZATION ======================*/
#define     ELEM_RAND_RANGE                 (100)
#define     ELEM_INIT_VALUE                 (1)

	/*=================== MACRO FUNCTION ======================*/
#define     PRINT_LINE(title)               printf("============== %s ==============\n", title)

	/*=================== EXECUTION MODE ======================*/
#define     MATRIX_MULT_CPU_ENABLE
#define     MATRIX_MULT_GPU_ENABLE
//#define     DONT_PRINT_MATRIX_FLAG

#include "../common/matop.h"



	int main(int argc, char *argv[]) {
		struct timeval start, end;
		double sum_duration, duration, gflops, gbps;
		ELEM_TYPE *a = NULL,
				  *b = NULL,
				  *c_h = NULL,
				  *c_d = NULL;
		char program_file[KERNEL_FILE_AND_FUNC_MAX_LEN],
			 kernel_func[KERNEL_FILE_AND_FUNC_MAX_LEN],
			 cl_build_program_options[KERNEL_FILE_AND_FUNC_MAX_LEN] = "-D ";
		int m = 0,
			n = 0,
			k = 0,
			len_a = 0,
			len_b = 0,
			len_c = 0,
			run_num = 0;
		size_t
			data_size_a = 0,
			data_size_b = 0,
			data_size_c = 0,
			global_work_size[3] = {1,1,1};

		if (argc == 10) {
			/*********************************
			  0. argc[0] file name
			  1. argc[1] m
			  2. argc[2] n
			  3. argc[3] k
			  4. argc[3] kernel_file_path
			  5. argc[4] kernel_func_name
			  6. argc[5] run_num
			  7. argc[6] global_work_size[0]
			  8. argc[7] global_work_size[1]
			  9. argc[8] global_work_size[2]
			*********************************/
			m = atoi( argv[1] );
			n = atoi( argv[2] );
			k = atoi( argv[3] );
			strcpy(program_file, argv[4]);
			strcpy(kernel_func, argv[5]);
			run_num = atoi( argv[6] );
			global_work_size[0] = atoi( argv[7] );
			global_work_size[1] = atoi( argv[8] );
			global_work_size[2] = atoi( argv[9] );
		   
		}
		else {
			printf(">>> [USAGE] %s M N K KERNEL_FILE_PATH KERNEL_FUNC_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]\n", argv[0]);
			printf(">>> [ERROR] please input args\n");
			exit(-1);
		}

		// check argc
		//for (int i = 0; i < 10; i++) printf("%d %s\n", i, argv[i]);

		strcat(cl_build_program_options, "CL_ELEM_TYPE=");
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
		a = (ELEM_TYPE *) malloc (data_size_a);
		b = (ELEM_TYPE *) malloc (data_size_b);
		c_h = (ELEM_TYPE *) malloc (data_size_c);
		c_d = (ELEM_TYPE *) malloc (data_size_c);
		
		printf(">>> [INFO] len_a: %d, len_b: %d, len_c: %d\n", len_a, len_b, len_c);
		printf(">>> [INFO] data_size_a: %d, data_size_b: %d, data_size_c: %d\n\n", (int)data_size_a, (int)data_size_b, (int)data_size_c);

    rand_mat(a, len_a, ELEM_RAND_RANGE);
    rand_mat(b, len_b, ELEM_RAND_RANGE);
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
    PRINT_LINE("CPU RESULT");
    printf(">>> [INFO] %d times %s starting...\n", run_num, "CPU");
    sum_duration = 0.0;
    for (int ridx; ridx < run_num; ridx++) {
        gettimeofday(&start, NULL);
        mult_mat(a, b, c_h, m, n, k);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) + 
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        sum_duration += duration;
    }
    gflops = 2.0 * m * n * k;
    gflops = gflops / sum_duration / (double)run_num *1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", "CPU", m, n, k, sum_duration, gflops);

#endif

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_h:\n");
    print_mat(c_h, n, m);
#endif

#ifdef MATRIX_MULT_GPU_ENABLE
    PRINT_LINE("GPU RESULT");

    cl_mem a_buffer = NULL,
           b_buffer = NULL,
           c_buffer = NULL;

    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms = 0;

    cl_device_id device_id = NULL;
    cl_uint ret_num_devices = 0;

    cl_context context = NULL;
    cl_kernel kernel = NULL;
    cl_program program = NULL;

    cl_command_queue command_queue = NULL;
    cl_event event = NULL;
    cl_int ret = 0;

    FILE *program_handle = NULL;
    char *program_buffer;
    size_t program_size;

    FILE *fp; char buffer[KERNEL_FILE_AND_FUNC_MAX_LEN];
    fp = popen("cat /sys/class/misc/mali0/device/gpuinfo", "r");
    char *ret_ = fgets(buffer, sizeof(buffer), fp);
    printf(">>> [INFO] Device name: %s", buffer);
    pclose(fp);

    // Load source
    printf(">>> [INFO] program_file: %s, kernel_func: %s\n", program_file, kernel_func);
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
    a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size_a, NULL, &ret);
    b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size_b, NULL, &ret);
    c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size_c, NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, a_buffer, CL_TRUE, 0, data_size_a, (void *)a, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, b_buffer, CL_TRUE, 0, data_size_b, (void *)b, 0, NULL, NULL);
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

    ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &m);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void *) &n);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &k);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &a_buffer);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &b_buffer);
    ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &c_buffer);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to set kernel arguments.%d\n", (int)ret);
        goto error;
    }

    printf(">>> [INFO] global_work_size[%d]: { %d, %d, %d }\n", GLOBAL_WORK_SIZE_DIM, (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2]);
    int global_size = (int) global_work_size[0] * (int) global_work_size[1] * (int) global_work_size[2];
    int task_size = m * n * k;
    if (global_size < task_size) {
        printf(">>> [WARN] global work size (%d) is smaller than task size (%d)\n", global_size, task_size);
    }

    /* GPU */
    printf(">>> [INFO] %s %d times %s.%s starting ...\n", OCL_DEVICE_TYPE, run_num, program_file, kernel_func);
    sum_duration = 0.0;
    for (int ridx = 0; ridx < (run_num+1); ridx++) {
        gettimeofday(&start, NULL);
        // Run kernel
        clEnqueueNDRangeKernel(command_queue, kernel, GLOBAL_WORK_SIZE_DIM, NULL, 
                               global_work_size,
                               LOCAL_WORK_SIZE_POINTER, 0, NULL, &event);
        clFinish(command_queue);
        gettimeofday(&end, NULL);
        duration = ((double)(end.tv_sec-start.tv_sec) + 
                    (double)(end.tv_usec-start.tv_usec)/1000000);
        if (ridx == 0) {
            printf(">>> [INFO] skip first time\n");
            continue;
        }
        sum_duration += duration;
    }
    gflops = 2.0 * m * n * k;
    gflops = gflops / sum_duration / (double)run_num *1.0e-9;
    gbps = 0;
    printf(">>> [INFO] %s %dx%dx%d %2.6lf s %2.6lf GFLOPS\n\n", "CPU", m, n, k, sum_duration, gflops);

    // Copy result from device to host
    ret = clEnqueueReadBuffer(command_queue, c_buffer, CL_TRUE, 0, data_size_c, (void *)c_d, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf(">>> [ERROR] failed to copy data from device to host: %d\n", (int)ret);
        goto error;
    }

    equal_vec(c_h, c_d, len_c);

#ifndef DONT_PRINT_MATRIX_FLAG
    printf("c_d:\n");
    print_mat(c_d, k, n);
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
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    free(program_buffer);

#endif

    free(a);
    free(b);
    free(c_h);
    free(c_d);

    return 1;
}
