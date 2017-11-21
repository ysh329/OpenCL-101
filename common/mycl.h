// OCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define    KERNEL_FILE_AND_FUNC_MAX_LEN    (100)

/*
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms = 0;

    cl_device_id device_id = NULL;
    cluint ret_num_device = 0;

    cl_context context = NULL;
    cl_kernel kernel = NULL;
    cl_program program = NULL;

    cl_command_queue command_queue = NULL;
    cl_event event = NULL;
    cl_int ret = 0;



    char program_file[KERNEL_FILE_AND_FUNC_MAX_LEN], 
         kernel_func[KERNEL_FILE_AND_FUNC_MAX_LEN],
         cl_build_program_options[KERNEL_FILE_AND_FUNC_MAX_LEN] = "-D ";
*/

void print_device(void) {

    FILE *fp; 
    char buffer[KERNEL_FILE_AND_FUNC_MAX_LEN];

    fp = popen("cat /sys/class/misc/mali0/device/gpuinfo", "r");
    char *ret_ = fgets(buffer, sizeof(buffer), fp);
    printf(">>> [INFO] Device name: %s", buffer);
    pclose(fp);
}


