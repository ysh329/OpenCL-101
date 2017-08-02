#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <time.h>
#include <sys/time.h>

/* gpu */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define 	MEM_SIZE 			(128)
#define 	MAX_SOURCE_SIZE 	(0x100000)

void init_vec(int *vec, int len, int set_one_flag) {
    for (int i = 0; i < len; i++) {
		if (set_one_flag)
			vec[i] = 1;
		else
			vec[i] = 0;
	}
}

void rand_vec(int *vec, int len) {
	srand( (unsigned) time(0) );
	for (int i = 0; i < len; i++) {
		vec[i] = rand() % 2;
	}
}

void add_vec_cpu(const int *a, const int *b, int *res, const int len) {
	for (int i = 0; i < len; i++) {
		res[i] = a[i] + b[i];
	}
}

void print_vec(int *vec, int len) {
	for (int i = 0; i < len; i++) {
		printf("%d ", vec[i]);
	}
	printf("\n");
}

void check_result(int *v1, int *v2, int len) {
    int correct_num = 0;
	for (int i = 0; i < len; i++) {
		if (v1[i] == v2[i]) {
			correct_num += 1;
		}
	}
	printf("correct rate: %d / %d , %1.2f\n", correct_num, len, (float)correct_num/len);
}

int main(void) {
	struct timeval start, finish;
	double duration;
	srand( (unsigned) time(NULL) );

	/* generate vector a and b */
	int len = 64;
	int *a, *b, *c, *c_d;
	a = (int *) malloc (len * sizeof(int));
	b = (int *) malloc (len * sizeof(int));
	c = (int *) malloc (len * sizeof(int));
	c_d = (int *) malloc (len * sizeof(int));
	size_t data_size = len * sizeof(int);

	/* vector addition, cpu version */
	printf("a: ");
	init_vec(a, len, 1);
	print_vec(a, len);

	printf("b: ");
	rand_vec(b, len);
	print_vec(b, len);

	printf("c: ");
	init_vec(c, len, 0);
	add_vec_cpu(a, b, c, len);
	print_vec(c, len);

	/* vector addition, gpu version  */
	cl_mem a_buff, b_buff, c_buff;
	a_buff = b_buff = c_buff = NULL;

	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platforms;

	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;

	cl_context context = NULL;
	cl_kernel kernel = NULL;
	cl_program program = NULL;

	cl_command_queue command_queue = NULL;
	cl_int ret;

	/* Load the source code containing the kernel */
	char string[MEM_SIZE];
	FILE *fp;
	char fileName[] = "./vec-add-standard.cl";
	char *source_str;
	size_t source_size;

	fp = fopen(fileName, "r");
	if (!fp) {
		
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*) malloc (MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Platform
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) {
	    printf("Failed to get platform ID.\n");
		goto error;
	}
	// Device
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (ret != CL_SUCCESS) {
	    printf("Failed to get device ID.\n");
		goto error;
	}
	// Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);//&ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create OpenCL context.\n");
	    goto error;
	}
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if (ret != CL_SUCCESS) {
	   printf("Failed to create command queue %d\n", (int) ret);
	   goto error;
	}
	// Memory Buffer
	a_buff = clCreateBuffer (context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	b_buff = clCreateBuffer (context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	c_buff = clCreateBuffer (context, CL_MEM_WRITE_ONLY, data_size, NULL, &ret);
	
	ret = clEnqueueWriteBuffer (command_queue, a_buff, CL_TRUE, 0, data_size, (void *)a, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, b_buff, CL_TRUE, 0, data_size, (void *)b, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to copy date from host to device: %d\n", (int) ret);
		goto error;
	}
	// Create Kernel Program from source
	 program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
			(const size_t *)&source_size, &ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create OpenCL program from source %d\n", (int) ret);
		goto error;
	}
	// Build Kernel Program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to build program %d\n", (int) ret);
		char build_log[16348];
		clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, sizeof (build_log), build_log, NULL);
		printf ("Error in kernel: %s\n", build_log);
		goto error;
	}
	// Create OpenCL Kernel
	kernel = clCreateKernel(program, "add_vec_gpu", &ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create kernel %d\n", (int) ret);
		goto error; 
	}
	ret  = clSetKernelArg(kernel, 0, sizeof (cl_mem), (void *) &a_buff);
	ret |= clSetKernelArg(kernel, 1, sizeof (cl_mem), (void *) &b_buff);
	ret |= clSetKernelArg(kernel, 2, sizeof (cl_mem), (void *) &c_buff);
	ret |= clSetKernelArg(kernel, 3, sizeof (cl_int), (void *) &len);
	if (ret != CL_SUCCESS) {
		printf("Failed to set kernel arguments %d\n", (int) ret);
		goto error;
	}

	/* Execute OpenCL Kernel */
	// executed using a single work-item
	// ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

    size_t global_work_size, local_work_size;  
    // Number of work items in each local work group  
    local_work_size = len;  
    // Number of total work items - localSize must be devisor  
    global_work_size = (size_t) ceil( len / (float) local_work_size ) * local_work_size;

	//size_t local_work_size[2] = { 8, 8 };
	//size_t global_work_size[2] = { 1, len };
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to execute kernel for execution %d\n", (int) ret);
		goto error;
	}

	init_vec(c_d, len, 0);
	/* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, c_buff, CL_TRUE, 0, data_size, (void *)c_d, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to copy data from device to host %d\n", (int) ret);
		goto error;
	}

	/* Display Result */
	printf("c: ");
	print_vec(c_d, len);
	check_result(c, c_d, len);
	printf("len-1=%d, c_d[%d]==c[%d]: %d, c_d[%d]=%d, c[%d]=%d \n", len-1, len-1, len-1, c_d[len-1]==c[len-1], len-1, c_d[len-1], len-1, c[len-1]);

	printf("idx  c  c_d\n");
	for(int i = 0; i < len; i++) {
		printf("%2d %2d %2d \n", i, c[i], c_d[i]);
	}

	/* Finalization */
error:

    /* free device resources */
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseMemObject(a_buff);
	clReleaseMemObject(b_buff);
	clReleaseMemObject(c_buff);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

    /* free host resources */
	free(source_str);
	free(a);
	free(b);
	free(c);

	return 0;
}
