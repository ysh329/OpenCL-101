#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <time.h>
#include <sys/time.h>

/* opencl */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define		MATRIX_TRANSPOSE_CPU_ENABLE
#define		MATRIX_TRANSPOSE_GPU_ENABLE

#define 	ELEM_RAND_RANGE		(100)
#define     MATRIX_WIDTH        (1024)
#define     MATRIX_HEIGHT       (MATRIX_WIDTH)
#define 	NDIM				(2)
#define 	RUN_NUM				(100)

#define 	NOT_PRINT_FLAG

#define 	KERNEL_FUNC			"matrixTranspose"

#if (NDIM == 1)
	#define 	PROGRAM_FILE			"matrixTranspose_v1.cl"
#elif (NDIM == 2)
	#define		PROGRAM_FILE			"matrixTranspose_v2.cl"
#elif (NDIM == 3)
	#define 	PROGRAM_FILE 			"matrixTranspose_v2.cl"
#else
#endif

#define     LOCAL_WORK_SIZE_P		NULL

#include "matop.h"

int main(int argc, char * argv[]) {

	struct timeval start, end;
	double duration, gflops;

    float 
		*a, 
		*a_T_cpu, 
		*a_T_gpu;

	int 
		heightA,  widthA,
		heightAT, widthAT,
		len,
		ndim,
		run_num;

	size_t 
		data_size;

	char
		program_file[] = PROGRAM_FILE;

/*********************************************************
1. argc[1] ndim 			default: 2
2. argc[2] heightA			default: 1024
3. argc[3] widthA			default: =heightA
4. argc[4] kernel_file_path	default: matrixTranspose_v1.cl
5. argc[5] run_num			default: 100
**********************************************************/
	if (argc == 3) {
		ndim = atoi( argv[1]  );
		heightA = atoi( argv[2] );
		widthA = heightA;
	}
	else if (argc == 4) {
		ndim = atoi( argv[1]  );
		heightA = atoi( argv[2] );
		widthA = atoi( argv[3] );
	}
	else if (argc == 5) {
		ndim = atoi( argv[1] );
		heightA = atoi( argv[2] );
		widthA = atoi( argv[3] );
		strcpy( program_file, argv[4] );
	}
	else if (argc == 6) {
		ndim = atoi( argv[1] );
		heightA = atoi( argv[2] );
		widthA = atoi( argv[3] );
		strcpy( program_file, argv[4] );
		run_num = atoi( argv[5] );
	}
	else {
		ndim = NDIM;
		heightA = MATRIX_HEIGHT; 
		widthA = MATRIX_WIDTH;
		// program_file was set previously
		run_num = RUN_NUM;
	}

	len = heightA * widthA;
	heightAT = widthA;
	widthAT = heightA;	
 
	data_size = heightA * widthA * sizeof(float);
    a       = (float *) malloc (data_size);
	a_T_cpu = (float *) malloc (data_size);
	a_T_gpu = (float *) malloc (data_size);

#ifndef NOT_PRINT_FLAG
    printf("a:\n");
#endif

    rand_mat(a, widthA*heightA, ELEM_RAND_RANGE);

#ifndef NOT_PRINT_FLAG
    print_mat(a, widthA, heightA);
	printf("a^T_CPU:\n");
#endif

#ifdef MATRIX_TRANSPOSE_CPU_ENABLE
	printf(">>> Starting %d CPU runs ...\n", run_num);
	gettimeofday(&start, NULL);
	for (int ridx = 0; ridx < run_num; ridx++)
		transpose_mat(a, widthA, heightA, a_T_cpu);
    gettimeofday(&end, NULL);
    duration = ((double)(end.tv_sec-start.tv_sec)*1000000 + 
        (double)(end.tv_usec-start.tv_usec)) / 1000000 / (double) run_num;
    gflops = 1.0 * heightA * widthA;
    gflops = gflops / duration * 1.0e-6;
    printf("CPU \t\t\t%d x %d\t%lf s\t%lf MFLOPS\n\n", widthA, heightA, duration, gflops);
#endif

#ifndef NOT_PRINT_FLAG
	print_mat(a_T_cpu, widthAT, heightAT);
	printf("set a^T_GPU all zero:\n");
#endif

	/* GPU */
	init_mat(a_T_gpu,  widthAT * heightAT, 0);

#ifndef NOT_PRINT_FLAG
	print_mat(a_T_gpu, widthAT, heightAT);
#endif

#ifdef MATRIX_TRANSPOSE_GPU_ENABLE
	cl_mem a_buff, a_T_buff;
	a_buff = a_T_buff = NULL;

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
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char *)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	// Platform
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);	
	if (ret != CL_SUCCESS) {
		printf("failed to get platform ID.\n");
		goto error;
	}

	// Device
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (ret != CL_SUCCESS) {
		printf("failed to get device id.\n");
		goto error;
	}

    // Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create OpenCL context.\n");
        goto error;
    }
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create command queue.\n");
        goto error;
    }
    // Memory Buffer
    a_buff   = clCreateBuffer (context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	a_T_buff = clCreateBuffer (context, CL_MEM_WRITE_ONLY, data_size, NULL, &ret);

    ret  = clEnqueueWriteBuffer (command_queue, a_buff,   CL_TRUE, 0, data_size, (void *)a,       0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, a_T_buff, CL_TRUE, 0, data_size, (void *)a_T_gpu, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to copy data from host to device.\n");
        goto error;
    }

    // Create Kernel Program from source
    program = clCreateProgramWithSource(context, 1, (const char **)&program_buffer,
				(const size_t *)&program_size, &ret);
    if (ret != CL_SUCCESS) {
        printf("failed to create OpenCL program from source %d\n", (int)ret);
        goto error;
    }

    // Build Kernel Program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
		printf("failed to build program %d\n", (int) ret);
		char build_log[16348];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
		printf("Error in kernel: %s\n", build_log);
		goto error;
    }
		
	// Create OpenCL kernel
	kernel = clCreateKernel(program, KERNEL_FUNC, &ret);
	if (ret != CL_SUCCESS) {
		printf("failed to create kernel %d\n", (int) ret);	
		goto error;
	}

	ret  = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &heightA);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void *) &widthA);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &a_buff);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &a_T_buff);
    if (ret != CL_SUCCESS) {
		printf("failed to set kernel arguments %d\n", (int) ret);
		goto error;
	}

	// local_work_size: Number of work items in each local work-group
	// global_work_size: Number of total work-items - localSize must be devisor
	size_t *global_work_size_pointer = NULL;
	if (ndim == 1) {
		size_t global_work_size = max(heightA, widthA);
		global_work_size_pointer = &global_work_size;
		printf(">>> global_work_size: %d\n", (int)global_work_size);
	}
	else if (ndim == 2) {
		size_t global_work_size[2] = { max(heightA, widthA), max(heightA, widthA) };
		global_work_size_pointer = global_work_size;
		printf(">>> global_work_size[%d]: (%d, %d)\n", ndim, (int)global_work_size[0], (int)global_work_size[1]);
	}
	else if (ndim == 3) {
		size_t global_work_size[3] = { max(heightA, widthA), max(heightA, widthA), 1};
		global_work_size_pointer = global_work_size;
		printf(">>> global_work_size[%d]: (%d, %d, %d)\n", ndim, (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2]);
	}
	else {
		printf("global_work_size set error.\n");
		goto error;
	}
		
	// Start the timed loop
	printf(">>> Starting %d %s runs ...\n", run_num, KERNEL_FUNC);
	gettimeofday(&start, NULL);
	for (int ridx = 0; ridx < run_num; ridx++) {
		// Run kernel
		clEnqueueNDRangeKernel(command_queue, kernel, ndim, NULL, global_work_size_pointer,//&global_work_size,
															   LOCAL_WORK_SIZE_P,//&local_work_size,
															   0, NULL, &event);
		clWaitForEvents(1, &event);
	}
	gettimeofday(&end, NULL);
    duration = ((double)(end.tv_sec-start.tv_sec)*1000000 + 
        (double)(end.tv_usec-start.tv_usec)) / 1000000 / (double) run_num;
    gflops = 1.0 * heightA * widthA;
    gflops = gflops / duration * 1.0e-6;
    printf("GPU %s %d x %d\t%lf s\t%lf MFLOPS\n\n", program_file, widthA, heightA, duration, gflops);

	
	// Copy the output matrix ( transposed matrix a_T_gpu ) result from the GPU memory ( a_T_buff )
	ret = clEnqueueReadBuffer(command_queue, a_T_buff, CL_TRUE, 0, data_size, (void *)a_T_gpu, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("failed to copy data from device to host %d\n", (int) ret);
		goto error;
	}
#endif

	// Display result
#ifndef NOT_PRINT_FLAG
	printf("a^T_GPU:\n");
	print_mat(a_T_gpu, widthAT, heightAT);
#endif

	// Check result
	//equal_mat(a_T_cpu, a_T_gpu, widthAT, heightAT);
	equal_vec(a_T_cpu, a_T_gpu, widthAT*heightAT);
	
error:
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	// Free the OpenCL memory objects
	clReleaseMemObject(a_buff);
	clReleaseMemObject(a_T_buff);

	// Clean-up OpenCL
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	clReleaseProgram(program);
    clReleaseKernel(kernel);

	// Free the host memory objects
	free(program_buffer);
	free(a);
	free(a_T_cpu);
	free(a_T_gpu);

	// Exit
    return 0;
}

