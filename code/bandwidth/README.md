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
$ ./bandwidth 1024 1024 ./bandwidth_kernel.cl global_bandwidth_float_v1 100 $[1024*1024] 1 1 
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


## TODO
- [ ] Support to monitor and set GPU, CPU frequency max;
- [ ] Support halfN precision;
- [ ] Support variable-type input from command line: 1. from cmd line to bandwidth.c; 2. from host to device (cl_program_build_options).
