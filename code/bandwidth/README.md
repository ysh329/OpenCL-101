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
$ # usage: ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUN_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 1024 1024 ./kernel.cl global_bandwidth_vec1 1 $[1024*1024] 1 1
>>> [INFO] ELEM_TYPE_STR: float
>>> [INFO] CL_ELEM_TYPE_STR: float
>>> [INFO] cl_program_build_options: -D CL_ELEM_TYPE=float
============== CPU RESULT ==============
>>> [INFO] 1 times CPU starting...
>>> [INFO] CPU 1024 x 1024 0.001900 s 1103.764211 MFLOPS

>>> [INFO] bandwidth: 4.11 GB/s
>>> [TEST] correct rate: 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b

============== GPU RESULT ==============
Mali-T86x MP4 r2p0 0x0860
>>> [INFO]program_file: ./kernel.cl
>>> [INFO] kernel_func: global_bandwidth_vec1
>>> [INFO] global_work_size[3]: { 1048576, 1, 1 }
>>> [INFO] 1 times ./kernel.cl.global_bandwidth_vec1 starting...
>>> [INFO] skip first time.
>>> [INFO] CL_GPU 1024 x 1024 0.002527 s 829.897903 MFLOPS ./kernel.cl

>>> [INFO] bandwidth: 3.09 GB/s
>>> [TEST] correct rate: 1.0000
>>> [TEST] ~ Bingo ~ matrix a == matrix b
```

## Test

This test script executes `./bandwidth` many times to test different matrix sizes for different kernels (such as floatN or intN).

```python
python benchmark.py
```

## Monitor

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
