# bandwidth

Compute bandwidth for GPU and CPU at different matrix sizes.

## Usage

### Bandwidth for float

Change these lines about type definations in [bandwidth.c](./bandwidth.c) as below:

```cc
#define   ELEM_TYPE                     float
#define   ELEM_TYPE_STR                 "float"
#define   CL_ELEM_TYPE                  cl_float
#define   CL_ELEM_TYPE_STR              "float"
```

And then, build `bandwidth.c` using following command:

```shell
$ ./make.sh
```

If build's okay, you can execute `./bandwidth` first to check how to use:

```shell
$ ./bandwidth 
>>> [USAGE] ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUNC_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]

============== INIT ==============
>>> [INFO] ELEM_TYPE_STR: float, sizeof(ELEM_TYPE): 4
>>> [INFO] CL_ELEM_TYPE_STR: float, sizeof(CL_ELEM_TYPE): 4
>>> [INFO] cl_program_build_options: -D CL_ELEM_TYPE=float -D CL_INPUT_TYPE=float
>>> [ERROR] please input args
```

Assuming a 2048\*2048 matrix, preparing-execution kernel file is `kernel.cl`, preparing-execution function is `global_bandwidth_vec1` (`vec1` representing non-vectorization) and run `100` times to measure bandwidth, global work size is (2048\*2048, 1, 1): 

```shell
# >>> [USAGE] ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUNC_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec1 100 $[2048*2048] 1 1
```

### Bandwidth for float8

Change these lines in `bandwidth.c` first:

```cc
#define   ELEM_TYPE                     float
#define   ELEM_TYPE_STR                 "float"
#define   CL_ELEM_TYPE                  cl_float8
#define   CL_ELEM_TYPE_STR              "float8"
```

Build `bandwidth.c`:

```shell
$ ./make.sh
```

Execute binary file as below, and watch out the difference between `float` and `float8`:
```shell
# >>> [USAGE] ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUNC_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec8 100 $[2048*2048/8] 1 1
```

### Bandwidth for half

Change these lines in `bandwidth.c`:

```cc
#define   ELEM_TYPE                     __fp16
#define   ELEM_TYPE_STR                 "__fp16"
#define   CL_ELEM_TYPE                  cl_half 
#define   CL_ELEM_TYPE_STR              "half"
```

Build:

```shell
$ ./make.sh
```

Execute:

```shell
$ # >>> [USAGE] ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUNC_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec1 100 $[2048*2048] 1 1
```

### Bandwidth for half16

Change these lines in `bandwidth.c`:

```cc
#define   ELEM_TYPE                     __fp16
#define   ELEM_TYPE_STR                 "__fp16"
#define   CL_ELEM_TYPE                  cl_half 
#define   CL_ELEM_TYPE_STR              "half16"
```

Note: The `half` type is an outlier (`halfN` is not defined in host: `cl_half2`, `cl_half4`, `cl_half8` and `cl_half16` are not in `/usr/include/CL/cl.h`). If define `CL_ELEM_TYPE` as `cl_half16` or `halfN`, it'll cause a problem: `cl_halfN is undeclared`.

Build:

```shell
$ ./make.sh
```

Execute:

```shell
$ # >>> [USAGE] ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUNC_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec16 100 $[2048*2048/16] 1 1
```

## Frequency Monitor & Modification

Please refer to [tools](../tools) directory.
