#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>
#include <sys/time.h>

/* opencl */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define 	ELEM_RAND_RANGE		100
#define     MATRIX_WIDTH        (2048)
#define     MATRIX_HEIGHT       (2048)
#define 	NDIM				(1)
#define 	RUN_NUM				(100)

#define 	NOT_PRINT_FLAG

#define 	KERNEL_FUNC			"matrixTranspose"

#if (NDIM == 1)
	#define 	PROGRAM_FILE			"matrixTranspose_v1.cl"
	#define 	SET_GLOBAL_WORK_SIZE	size_t global_work_size = MATRIX_WIDTH;
	#define 	PRINT_GLOBAL_WORK_SIZE	printf(">>> global_work_size: %d\n", (int)global_work_size);
	#define 	GLOBAL_WORK_SIZE_P		&global_work_size
#elif (NDIM == 2)
	#define		PROGRAM_FILE			"matrixTranspose_v2.cl"
	#define 	SET_GLOBAL_WORK_SIZE    size_t global_work_size[NDIM] = {MATRIX_WIDTH, MATRIX_HEIGHT};
	#define 	PRINT_GLOBAL_WORK_SIZE	printf(">>> global_work_size[%d]: (%d, %d)\n", NDIM, (int)global_work_size[0], (int)global_work_size[1]);
	#define 	GLOBAL_WORK_SIZE_P		global_work_size
#elif (NDIM == 3)
	#define 	PROGRAM_FILE 			"matrixTranspose_v2.cl"
	#define 	SET_GLOBAL_WORK_SIZE	size_t global_work_size[NDIM] = {MATRIX_WIDTH, MATRIX_HEIGHT,1};
	#define 	PRINT_GLOBAL_WORK_SIZE	printf(">>> global_work_size[%d]: (%d, %d, %d)\n", NDIM, (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2]);
	#define		GLOBAL_WORK_SIZE_P		global_work_size
#else
#endif


void init_mat(float *mat, int len, float setVal) {
    for (int idx = 0; idx < len; idx++)
		mat[idx] = setVal;
}

void rand_mat(float *mat, int len, int range) {
    if (range < 1) {
        printf("range value can't be less than 1.\n");
        exit(-1);
    }
    srand( (unsigned) time(0) );
    for (int idx = 0; idx < len; idx++)
        mat[idx] = rand() % range;
}

void print_mat(float *mat, int width, int height) {
#ifdef NOT_PRINT_FLAG
	return;
#endif
	for (int r = 0; r < height; r++) {
	    for (int c = 0; c < width; c++) 
            printf("%.2f ", mat[c*height+r]);
        printf("\n");
    }
    printf("\n");
}

void print_vec(float *vec, int len) {
	for (int idx = 0; idx < len; idx++)
		printf("%.2f \n", vec[idx]);
}

void add_mat(float *a, float *b, float *res, int width, int height) {
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++)
            res[c*height + r] = a[c*height + r] + b[c*height + r];
}

float max(float a, float b) {
	if (a > b)
		return a;
	else
		return b;
}

void add_vec(float *a, float *b, float *res, int len) {
    for (int idx = 0; idx < len; idx++) 
        res[idx] = a[idx] + b[idx];
}

void transpose_mat(float *a, int width, int height, float *res) {
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++) {
            res[r*width + c] = a[c*height + r];
			//printf("res[%d, %d]:= a[%d, %d] = %.2f \n", r, c, c, r, a[c*height+r]);
			//printf("res[%d] := a[%d] = %.2f \n\n", r*width + c, c*height + r, a[c*height+r]);
		}
}

int equal_mat(float *a, float *b, int width, int height) {
	int correct_num = 0;
	for (int c = 0; c < width; c++)
		for (int r = 0; r < height; r++) 
			if (a[c*height+r] - b[c*height+r] < 10e-8) 
				correct_num += 1;

	float correct_rate = (float) correct_num / ( width * height );
	printf(">>> correct rate: %.4f\n", correct_rate);
	if (1.0 - correct_rate < 10e-6)
		printf(">>> ~ Bingo ~ matrix a == matrix b\n");
	else
		printf(">>> matrix a is equal to matrix b\n");
	return 1;
}

int equal_vec(float *a, float *b, int len) {
	int correct_num = 0;
	for (int idx = 0; idx < len; idx++) 
		if (a[idx] - b[idx] < 10e-8) 
			correct_num += 1;

	float correct_rate = (float) correct_num / len;
	printf(">>> correct rate: %.4f\n", correct_rate);
	if (1.0 - correct_rate < 10e-8)
		printf(">>> ~ Bingo ~ matrix a == matrix b\n");
	else
		printf(">>> matrix a is NOT equal to matrix b\n");
	return 1;
}

void dotprod_mat(float *a, float *b, float *res, int len) {
	for (int idx; idx < len; idx++)
		res[idx] = a[idx] * b[idx];
}

int mult_mat(float *a, float *b, float *res, int M, int N, int K) {
	int i, j, p;
	init_mat(res, M*K, 0);
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++) 
			for (p = 0; p < K; p++) {
				res[j * M + i] += a[p * M + i] * b[j * K + p];
				printf("res[%d, %d] %.2f += a[%d, %d] %.2f * b[%d, %d] %.2f \n", i, j, res[j*M+i], i, p, a[p*M+i], p, j, b[j*K+p]);
			}
}


int main(void) {

	struct timeval start, end;
	double duration;

    float 
		*a, 
		*a_T_cpu, 
		*a_T_gpu;

	int 
		heightA,  widthA,
		heightAT, widthAT,
		len;

	size_t 
		data_size;

	heightA = MATRIX_HEIGHT, 
	widthA = MATRIX_WIDTH,
	len = MATRIX_HEIGHT * MATRIX_WIDTH;

	heightAT = widthA;
	widthAT = heightA;	
 
	data_size = heightA * widthA * sizeof(float);
    a       = (float *) malloc (data_size);
	a_T_cpu = (float *) malloc (data_size);
	a_T_gpu = (float *) malloc (data_size);

    printf("a:\n");
    rand_mat(a, widthA*heightA, ELEM_RAND_RANGE);
    print_mat(a, widthA, heightA);

	printf("a^T_CPU:\n");
	transpose_mat(a, widthA, heightA, a_T_cpu);
	print_mat(a_T_cpu, widthAT, heightAT);

	printf("set a^T_GPU all zero:\n");
	init_mat(a_T_gpu,  widthAT * heightAT, 0);
	print_mat(a_T_gpu, widthAT, heightAT);

	/* GPU */
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

	program_handle = fopen(PROGRAM_FILE, "r");
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

	//size_t global_work_size, local_work_size;
	//local_work_size = len; // Number of work items in each local work-group
	// Number of total work-items - localSize must be devisor
	//size_t global_work_size = (size_t) max(widthA, heightA);
	SET_GLOBAL_WORK_SIZE
	PRINT_GLOBAL_WORK_SIZE
	
	/*
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to execute kernel for execution %d\n", (int) ret);
        goto error;
    }
	*/

	// Start the timed loop
	printf(">>> Starting %d %s runs ...\n", RUN_NUM, KERNEL_FUNC);
	gettimeofday(&start, NULL);
	for (int ridx = 0; ridx < RUN_NUM; ridx++) {
		// Run kernel
		clEnqueueNDRangeKernel(command_queue, kernel, NDIM, NULL, GLOBAL_WORK_SIZE_P,//&global_work_size,
															   NULL,// &local_work_size,
															   0, NULL, &event);
		clWaitForEvents(1, &event);
	}
	gettimeofday(&end, NULL);
    duration = ((double)(end.tv_sec-start.tv_sec)*1000000 + 
        (double)(end.tv_usec-start.tv_usec)) / 1000000 / (double) RUN_NUM;
    double gflops = 1.0 * heightA * widthA;
    gflops = gflops / duration * 1.0e-6;
    printf("%s.%s \n %d x %d\t%lf s\t%lf MFLOPS\n\n", PROGRAM_FILE, KERNEL_FUNC, widthA, heightA, duration, gflops);

	
	// Copy the output matrix ( transposed matrix a_T_gpu ) result from the GPU memory ( a_T_buff )
	ret = clEnqueueReadBuffer(command_queue, a_T_buff, CL_TRUE, 0, data_size, (void *)a_T_gpu, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("failed to copy data from device to host %d\n", (int) ret);
		goto error;
	}

	// Display result
	printf("a^T_GPU:\n");
	print_mat(a_T_gpu, widthAT, heightAT);

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

