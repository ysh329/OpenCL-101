# bandwidth

Compute bandwidth for GPU and CPU at different matrix sizes.

## Build

```shell
./make.sh
```

## Usage

After build, you can execute binary file `./bandwidth` and append other parameters as below: 

```shell
$ # usage: ./bandwidth HEGHTA WIDTHA KERNEL_FILE_PATH KERNEL_FUN_NAME LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./bandwidth 1024 1024 ./kernel.cl global_bandwidth_float_v1 100 $[1024*1024] 1 1 
============== CPU RESULT ==============
>>> 100 times CPU starting...
>>> CPU 1024 x 1024 0.001524 s 1376.282665 MFLOPS

>>> bandwidth: 5.13 GB/s
>>> correct rate: 1.0000
>>> ~ Bingo ~ matrix a == matrix b

============== GPU RESULT ==============
Mali-T86x MP4 r2p0 0x0860
>>> program_file: ./bandwidth_kernel.cl
>>> kernel_func: global_bandwidth_float_v1
>>> global_work_size[3]: { 1048576, 1, 1 }

>>> 100 times ./bandwidth_kernel.cl.global_bandwidth_float_v1 starting...
>>> [NOTE] skip first time.
>>> CL_GPU 1024 x 1024 0.002519 s 425.731222 MFLOPS ./bandwidth_kernel.cl

>>> bandwidth: 3.10 GB/s
>>> correct rate: 1.0000
>>> ~ Bingo ~ matrix a == matrix b

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
