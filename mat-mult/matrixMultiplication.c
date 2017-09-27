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
#define 	ELEM_TYPE						float
#define 	ELEM_TYPE_STR       			"float"
// Type on Device (GPU)
#define     CL_ELEM_TYPE					cl_float
#define 	CL_ELEM_TYPE_STR 				"float"

/*=================== OTHER OCL PARAMETERS ================*/
#define 	CL_OTHER_MACRO					" -cl-mad-enable"
#define     OCL_DEVICE_TYPE					"CL_GPU" // "CL_CPU" or "CL_GPU"
#define     LOCAL_WORK_SIZE_POINTER			NULL
#define 	KERNEL_FILE_AND_FUNC_NAME_LEN  (100)

/*=================== INITIALIZATION ======================*/
#define 	ELEM_RAND_RANGE					(100)
#define 	ELEM_INIT_VALUE					(1)

/*=================== MACRO FUNCTION ======================*/
#define     PRINT_LINE(title)               printf("============== %s ==============\n", title)

/*=================== EXECUTION MODE ======================*/
#define     MATRIXMULT_CPU_ENABLE
#define     MATRIXMULT_GPU_ENABLE
#define     DONT_PRINT_RESULT_FLAG

#include "../common/matop.h"



int main(int argc, char *argv[]) {
    struct timeval start, end;
    double duration, gflops, gbps;
    ELEM_TYPE *a = NULL,
              *b = NULL,
              *c_from_h = NULL,
              *c_from_d = NULL;
    char program_file[KERNEL_FILE_AND_FUNC_NAME_LEN],
         kernel_func[KERNEL_FILE_AND_FUNC_NAME_LEN],
         cl_build_program_options[KERNEL_FILE_AND_FUNC_NAME_LEN];
    int m = 0,
        n = 0,
        k = 0,
        run_num = 0,
        ndim = 3;
    size_t
        data_size = 0,
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
    }

    // check argc
    //for (int i = 0; i < 10; i++) printf("%d %s\n", i, argv[i]);

    cl_build_program_options[KERNEL_FILE_AND_FUNC_MAX_LEN] = "-D ";
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
        printf(">>> [ERROR] CL_INPUT_TYPE defination is wrong.\n");
        exit(-1);
    }

}
