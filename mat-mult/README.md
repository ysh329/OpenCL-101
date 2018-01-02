# gemm-opt

## Setup

Edit `matMultWithInterleaveTrans.c` first:
```shell
vim matMultWithInterleaveTrans.c
```

Define `TYPE`, `VEC_LEN` and `USE_LOCAL_WORK_SIZE`:
```cc
/*================= CHANGE TYPE HERE ==================*/

// 8 or 4
#define         VEC_LEN                          (4)
#define         VEC_LEN_STR                      "4"

// FP32 or FP16
#define         FP32

// comment or uncomment
#define         USE_LOCAL_WORK_SIZE
```

## Build

```shell
./makeMatMultWithInterleaveTrans.sh
```

## Usage & Run

Check usage first using `./matMultWithInterleaveTrans`:

```shell
$ ./matMultWithInterleaveTrans 
>>> [USAGE] ./matMultWithInterleaveTrans M N K \
            TRANS_KERNEL_PATH      TRANS_KERNEL_FUNC_NAME       GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2] \
            INTERLEAVE_KERNEL_PATH INTERLEAVE__KERNEL_FUNC_NAME GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2] \
            MULT_KERNEL_PATH       MULT_KERNEL_FUNC_NAME        GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2] \
            CPU_BENCHMARK_TIMES    GPU_BENCHMARK_TIMES
>>> [ERROR] please input args
```

According to usage, run an example of `FP32, VEC_LEN=4, M=N=1024, K=1020` gemm as below:
```shell
./matMultWithInterleaveTrans 1024 1024 1020 \
./gemm_interleave_trans.c mat_trans_vec4 1020 256 1 \
./gemm_interleave_trans.c mat_interleave_vec4  256 255 1 \
./gemm_interleave_trans.c gemm_interleaved_transposed_vec4 $[256/1] $[256/1] 1 \
1 10
```

Execution logs as below:
```shell
>>> [INFO] cl_build_program_options: -D  USE_LOCAL_WORK_SIZE -D VEC_LEN=4 -D CL_ELEM_TYPE=float4 -D CL_INPUT_TYPE=float
============= INIT ============
>>> [INFO] ELEM_TYPE_STR: float, sizeof(ELEM_TYPE): 4
>>> [INFO] CL_ELEM_TYPE_STR: float4, sizeof(CL_ELEM_TYPE): 4
============= CPU MATRIX MULTIPLICATION ============
>>> [INFO] 1 times CPU starting...
0        1.088785
>>> [INFO] skip first 1 time(s)
1        1.082583
>>> [INFO] CPU 1024x1024x1020 1.082583 s 1.975918 GFLOPS

============= CPU MATRIX TRANSPOSE ============
0        0.007497
>>> [INFO] skip first 1 time(s)
1        0.008638
>>> [INFO] CPU-MATRIX-TRANS-FOR-B 1020x1024 0.008638 s 0.120917 GFLOPS

============= GPU MATRIX TRANSPOSE ============
>>> [INFO] Device name: Mali-T86x MP4 r2p0 0x0860
>>> [INFO] global_work_size[3]: { 256, 1020, 1 }
>>> [WARN] global work size (261120) is smaller than task size (1048576)
>>> [INFO] cl_build_program_options: -D  USE_LOCAL_WORK_SIZE -D VEC_LEN=4 -D CL_ELEM_TYPE=float4 -D CL_INPUT_TYPE=float
>>> [INFO] CL_GPU 10 times ./gemm_interleave_trans.c.mat_trans_vec4 starting ...
0        0.005359
>>> [INFO] skip first 1 time(s)
1        0.002973
2        0.002860
3        0.002813
4        0.002821
5        0.002875
6        0.003519
7        0.003192
8        0.003052
9        0.002981
10       0.002906
>>> [INFO] CL_GPU 1020x1024 0.002999 s 0.348253 GFLOPS

============= GPU INTERLEAVE ============
>>> [INFO] global_work_size[3]: { 255, 256, 1 }
>>> [WARN] global work size (65280) is smaller than task size (1044480)
>>> [INFO] cl_build_program_options: -D  USE_LOCAL_WORK_SIZE -D VEC_LEN=4 -D CL_ELEM_TYPE=float4 -D CL_INPUT_TYPE=float
>>> [INFO] CL_GPU 10 times ./gemm_interleave_trans.c.mat_interleave_vec4 starting ...
0        0.004367
>>> [INFO] skip first 1 time(s)
1        0.003220
2        0.001873
3        0.001834
4        0.001823
5        0.001818
6        0.001853
7        0.001839
8        0.011816
9        0.002071
10       0.001875
>>> [INFO] CL_GPU 1024x1020 0.003002 s 0.347905 GFLOPS

============= GPU MATRIX MULTIPLICATION ============
>>> [INFO] aI_height: 256        bT_height: 256  bT_width/aI_width: 4080/4080 
>>> [INFO] global_work_size[3]: { 256, 256, 1 }>>> [INFO] local_work_size[3]: { 4, 4, 1 } 
>>> [WARN] global work size (1048576) > task size (65536)
>>> [INFO] cl_build_program_options: -D  USE_LOCAL_WORK_SIZE -D VEC_LEN=4 -D CL_ELEM_TYPE=float4 -D CL_INPUT_TYPE=float
>>> [INFO] CL_GPU 10 times ./gemm_interleave_trans.c.gemm_interleaved_transposed_vec4 starting ...
0        0.081768
>>> [INFO] skip first 1 time(s)
1        0.078778
2        0.077039
3        0.076913
4        0.076905
5        0.076884
6        0.077125
7        0.076610
8        0.076549
9        0.076703
10       0.076804
>>> [INFO] CL_GPU 1024x1024x1020 0.077031 s 27.769275 GFLOPS

>>> [TEST] correct rate(1048576/1048576): 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b
```
