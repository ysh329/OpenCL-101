# bandwidth

Compute bandwidth for GPU and CPU at different matrix sizes.

## Build

If you want to measure `bandwidth` of `float` type for cpu and gpu, you should change these lines as bellow in `bandwidth.c` before build:

```c
// CPU TYPE
#define   ELEM_TYPE                     float
#define   ELEM_TYPE_STR                 "float"
// GPU TYPE
#define   CL_ELEM_TYPE                  float
#define   CL_ELEM_TYPE_STR              "float"
```
After change, you can build `bandwidth.c` using following command:

```shell
./make.sh
```

## Usage

After build, you can execute binary file `./bandwidth` and append other parameters as below:



```shell
$ # measure float-type bandiwdth of 2048*2048 size
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec1 10 $[2048*2048] 1 1

$ # measure float2-type bandiwith of 2048*2048 size
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec2 10 $[2048*2048/2] 1 1
``` 

### float

The execution logs of **float-type** (Be sure `ELEM_TYPE(_STR)` and `CL_ELEM_TYPE(_STR)` in `bandwith.c` are right and remeber to execute `make.sh`) bandwith of 2048*2048 size:

```shell
$ # usage: ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUN_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec1 100 $[2048*2048] 1 1
>>> [INFO] ELEM_TYPE_STR: float
>>> [INFO] CL_ELEM_TYPE_STR: float
>>> [INFO] cl_program_build_options: -D CL_ELEM_TYPE=float
============== CPU RESULT ==============
>>> [INFO] 100 times CPU starting...
>>> [INFO] CPU 2048 x 2048 0.006397 s 1311.387988 MFLOPS

>>> [INFO] bandwidth: 4.89 GB/s
>>> [TEST] correct rate: 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b

============== GPU RESULT ==============
Mali-T86x MP4 r2p0 0x0860
>>> [INFO]program_file: ./kernel.cl
>>> [INFO] kernel_func: global_bandwidth_vec1
>>> [INFO] global_work_size[3]: { 4194304, 1, 1 }
>>> [INFO] 100 times ./kernel.cl.global_bandwidth_vec1 starting...
>>> [INFO] skip first time.
>>> [INFO] CL_GPU 2048 x 2048 0.008171 s 1356.282619 MFLOPS ./kernel.cl

>>> [INFO] bandwidth: 3.82 GB/s
>>> [TEST] correct rate: 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b

```

### float2

The execution logs of **float2-type** (Be sure `ELEM_TYPE(_STR)` and `CL_ELEM_TYPE(_STR)` in `bandwith.c` are right and remeber to execute `make.sh`) bandwith of 2048*2048 size:

```shell
$ # usage: ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUN_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 2048 2048 ./kernel.cl global_bandwidth_vec2 100 $[2048*2048/2] 1 1
>>> [INFO] ELEM_TYPE_STR: float
>>> [INFO] CL_ELEM_TYPE_STR: float2
>>> [INFO] cl_program_build_options: -D CL_ELEM_TYPE=float2
============== CPU RESULT ==============
>>> [INFO] 100 times CPU starting...
>>> [INFO] CPU 2048 x 2048 0.006483 s 1294.001101 MFLOPS

>>> [INFO] bandwidth: 4.82 GB/s
>>> [TEST] correct rate: 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b

============== GPU RESULT ==============
Mali-T86x MP4 r2p0 0x0860
>>> [INFO]program_file: ./kernel.cl
>>> [INFO] kernel_func: global_bandwidth_vec2
>>> [INFO] global_work_size[3]: { 2097152, 1, 1 }
>>> [WARN] global work size (2097152) is smaller than task size (4194304).

>>> [INFO] 100 times ./kernel.cl.global_bandwidth_vec2 starting...
>>> [INFO] skip first time.
>>> [INFO] CL_GPU 2048 x 2048 0.006278 s 1460.920933 MFLOPS ./kernel.cl

>>> [INFO] bandwidth: 4.98 GB/s
>>> [TEST] correct rate: 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b
```

## Test

This test script executes `./bandwidth` many times to test different matrix sizes for different kernels (such as floatN or intN).

```python
python benchmark.py
```

## Monitor


### CPU

monitor cpu frequency:
```shell
sudo cpufreq-aperf
```

If your CPU doesn't support measure, it will output `CPU doesn't support APERF/MPERF`. Otherwise, it'll output similar as below:

```shell
 CPU  Average freq(KHz) Time in C0  Time in Cx  C0 percentage
 000  4420000     00 sec 105 ms 00 sec 894 ms 10
 001  4318000     00 sec 123 ms 00 sec 876 ms 12
 002  4046000     00 sec 186 ms 00 sec 813 ms 18
 003  4250000     00 sec 020 ms 00 sec 979 ms 02
 004  4454000     00 sec 053 ms 00 sec 946 ms 05
 005  4522000     00 sec 141 ms 00 sec 858 ms 14
 006  4352000     00 sec 082 ms 00 sec 917 ms 08
 007  4556000     00 sec 247 ms 00 sec 752 ms 24
```

### GPU

```shell
/sys/class/misc/mali0/device/devfreq/ff9a0000.gpu
```
