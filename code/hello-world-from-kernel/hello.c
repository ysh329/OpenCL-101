#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
 
int main()
{
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
 
    char string[MEM_SIZE];
 
    FILE *fp;
    char fileName[] = "./hello.cl";
    char *source_str;
    size_t source_size;
 
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
    	fprintf(stderr, "Failed to load kernel.\n");
    	exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    printf("source_size: %d\n", (int)source_size);
    close(fp);
 
    /* Get Platform and Device Info */
    /*  cl_int clGetPlatformIDs(	cl_uint num_entries,
        cl_platform_id *platforms,
        cl_uint *num_platforms)
       input params:
         num_entries: The number of cl_platform_id entries that can be added to platforms.
	 Platform is NULL or int number, which's bigger than zero.
	*platforms: Returns a list of OpenCL platforms found. 
	The cl_platform_id values returned in platforms can be used to identify
         a specific OpenCL platform. 
						If platforms argument is NULL, this argument is ignored. 
						The number of OpenCL platforms returned is the mininum of the value
						 specified by num_entries or the number of OpenCL platforms available.

			num_platforms: Returns the number of OpenCL platforms available. 
						   If num_platforms is NULL, this argument is ignored.
    */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("platform_id: %ld\n", (int)platform_id);
    printf("ret: %s\n", (int)ret);
    printf("ret_num_platforms: %u\n", (int)ret_num_platforms);
    printf("CL_DEVICE_TYPE_DEFAULT: %d\n", CL_DEVICE_TYPE_DEFAULT);
    printf("device_id: %d\n", (int)device_id);
    printf("ret_num_devices: %d\n", (int)ret_num_devices);
 
	/* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    printf("context: %s\n", context);
 
    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    /* Create Memory Buffer */
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);
 
    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
    (const size_t *)&source_size, &ret);
    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "hello", &ret);
 
    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
 
    /* Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
 
    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
    MEM_SIZE * sizeof(char),string, 0, NULL, NULL);
 
    /* Display Result */
    puts(string);
 
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
	 
    free(source_str);
 
    return 0;
}
