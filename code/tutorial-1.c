#include <stdio.h>
#include <stdlib.h>

/* OpenCL Header file */
#include <CL/cl.h>

/* OpenCL Kernel function */
const char *program_source =
"__kernel void vec_add (__global int *a, __global int *b, __global int *c) { \n"
"    /* 获取任务项的唯一ID */ \n"
"    int idx = get_global_id (0); \n"
"    printf('%d ', idx); \n"
"    /* 将对应位置的 a 和 b 相加，结果存在 c 中 */ \n"
"    c[idx] = a[idx] + b[idx]; \n"
"} \n";

int main () {
    /* 主机上的数据 */
    int *a = NULL;
    int *b = NULL;
    int *c = NULL;

    /* 每个数组包含的元素数目 */
    const int num_elements = 2048;

    /* 数组实际大小 */
    size_t data_size = sizeof (int) * num_elements;

    /* 为数组申请内存空间 */
    a = (int *) malloc (data_size);
    b = (int *) malloc (data_size);
    c = (int *) malloc (data_size);

    /* 初始化数据 */
    for (int i = 0; i < num_elements; i++) {
        a[i] = i;
        b[i] = i;
    }

    /* 获取 OpenCL 平台的数目 */
    cl_uint  num_platforms = 0;
    clGetPlatformIDs (0, NULL, &num_platforms);
    printf("num_platforms: %d\n", num_platforms);

    /* 申请一段内存空间用来存放支持的平台信息 */
    cl_platform_id *platforms = (cl_platform_id *) malloc (num_platforms * sizeof (cl_platform_id));

    /* 获取平台信息 */
    clGetPlatformIDs (num_platforms, platforms, NULL);

    /* 获取平台上面的设备数目 */
    cl_uint num_devices = 0;
    clGetDeviceIDs (platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

    /* 申请一段内存空间来存放所选平台支持的设备信息 */
    cl_device_id *devices;
    devices = (cl_device_id *) malloc (num_devices * sizeof (cl_device_id));

    /* 获取设备信息 */
    clGetDeviceIDs (platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    /* 创建 OpenCL 上下文，并且将其与设备关联起来 */
    cl_context context = clCreateContext (NULL, num_devices, devices, NULL, NULL, NULL);

    /* 创建命令队列，并且将其与设备关联起来 */
    cl_command_queue cmd_queue = clCreateCommandQueue (context, devices[0], 0, NULL);

    /* 创建三个 buffer 对象，用来存放数组的数据 */
    cl_mem buffer_a, buffer_b, buffer_c;
    buffer_a = clCreateBuffer (context, CL_MEM_READ_ONLY, data_size, NULL, NULL);
    buffer_b = clCreateBuffer (context, CL_MEM_READ_ONLY, data_size, NULL, NULL);
    buffer_c = clCreateBuffer (context, CL_MEM_READ_WRITE, data_size, NULL, NULL);

    /* 将 a 和 b 两个数组的数据复制到 buffer_a 和 buffer_b */
    clEnqueueWriteBuffer (cmd_queue, buffer_a, CL_FALSE, 0, data_size, a, 0, NULL, NULL);
    clEnqueueWriteBuffer (cmd_queue, buffer_b, CL_FALSE, 0, data_size, b, 0, NULL, NULL);

    /* 从源码创建 OpenCL 程序 */
    cl_program program = clCreateProgramWithSource (context, 1, (const char **) &program_source, NULL, NULL);

    /* 为 OpenCL 设备构建（编译）程序 */
    clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);

    /* 创建矢量相加的 OpenCL 内核 */
    cl_kernel kernel = clCreateKernel (program, "vec_add", NULL);

    /* 向内核函数传递参数 */
    clSetKernelArg (kernel, 0, sizeof (cl_mem), &buffer_a);
    clSetKernelArg (kernel, 1, sizeof (cl_mem), &buffer_b);
    clSetKernelArg (kernel, 2, sizeof (cl_mem), &buffer_c);

    /* 定义索引空间（有多少个工作项），这里方便起见我们只用了一个工作组 */
    size_t global_work_size[1];

    /* 这一个工作组里面有 num_elements 这么多个工作项 */
    global_work_size[0] = num_elements;

    /* 执行内核计算 */
    clEnqueueNDRangeKernel (cmd_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    /* 将计算之后 buffer_c 的数据读取到数组 c 中 */
    clEnqueueReadBuffer (cmd_queue, buffer_c, CL_TRUE, 0, data_size, c, 0, NULL, NULL);

    /* 验证计算结果 */
    for (int i = 0; i < num_elements; i++) {
        if (c[i] != i + i) {
            printf ("Output is incorrect\n");
            break;
        }
    }
    printf ("Output is correct\n");

    /* 清理设备资源 */
    clReleaseKernel (kernel);
    clReleaseProgram (program);
    clReleaseCommandQueue (cmd_queue);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    clReleaseContext (context);

    /* 清理主机资源 */
    free (a);
    free (b);
    free (c);
    free (platforms);
    free (devices);

    return 0;
}
