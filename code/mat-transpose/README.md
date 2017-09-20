# mat-transpose

## Build

```shell
$ ./make.sh
```

## Usage

There're some usage examples below:

### float16 type

```shell
# using command: ./matrixTranspose HEIGHTA WIDTHA KERNEL_FILE_PATH LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./matrixTranspose 2048 2048 ./matrixTranspose_v3_float16.cl 1 $[2048/4] 2048 1
>>> 1 times CPU starting...
CPU 2048 x 2048 0.221085 s 18.971454 MFLOPS

>>> global_work_size[3]: (512, 2048, 1)
[WARN] global work size is smaller than task size.
>>> 1 times ./matrixTranspose_v3_float16.cl starting...
GPU 2048 x 2048 0.105491 s 39.759828 MFLOPS ./matrixTranspose_v3_float16.cl

>>> correct rate: 1.0000
>>> ~ Bingo ~ matrix a == matrix b
```
### float type

```shell
# using command: ./matrixTranspose HEIGHTA WIDTHA KERNEL_FILE_PATH LOOP_EXECUTION_TIMES GLOBAL_WORK_SIZE[0] GLOBAL_WORK_SIZE[1] GLOBAL_WORK_SIZE[2]
$ ./matrixTranspose 2048 2048 ./matrixTranspose_v1.cl 1 $[2048*2048] 1 1
>>> 1 times CPU starting...
CPU 2048 x 2048 0.220617 s 19.011699 MFLOPS

>>> global_work_size[3]: (4194304, 1, 1)
[WARN] global work size is smaller than task size.
[WARN] using kernel-v1, the second and third dim of global work size should be one.
>>> new global_work_size[3]: (4194304, 1, 1)
>>> 1 times ./matrixTranspose_v1.cl starting...
GPU 2048 x 2048 0.089062 s 47.094204 MFLOPS ./matrixTranspose_v1.cl

>>> correct rate: 1.0000
>>> ~ Bingo ~ matrix a == matrix b
```
