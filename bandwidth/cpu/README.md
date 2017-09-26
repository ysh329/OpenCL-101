# bandwidth-cpu

The core code (`stream.c`) is from [jeffhammond/STREAM: STREAM benchmark](https://github.com/jeffhammond/STREAM), whose excution result is as reference for opencl bandwidth.

## Build

```shell
./make.sh
```

## Usage

```shell
./stream
```

## Result

This is my execution logs on Firefly-RK3399 board: 

```shell
$ ./stream 
-------------------------------------------------------------
STREAM version $Revision: 5.10 $
-------------------------------------------------------------
This system uses 8 bytes per array element.
-------------------------------------------------------------
Array size = 10000000 (elements), Offset = 0 (elements)
Memory per array = 76.3 MiB (= 0.1 GiB).
Total memory required = 228.9 MiB (= 0.2 GiB).
Each kernel will be executed 10 times.
 The *best* time for each kernel (excluding the first iteration)
 will be used to compute the reported bandwidth.
-------------------------------------------------------------
Your clock granularity/precision appears to be 1 microseconds.
Each test below will take on the order of 77378 microseconds.
   (= 77378 clock ticks)
Increase the size of the arrays if this shows that
you are not getting at least 20 clock ticks per test.
-------------------------------------------------------------
WARNING -- The above is only a rough guideline.
For best results, please be sure you know the
precision of your system timer.
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:            2069.2     0.077694     0.077325     0.077805
Scale:           2348.2     0.068550     0.068136     0.068685
Add:             3273.5     0.073703     0.073315     0.073869
Triad:           3360.3     0.071474     0.071422     0.071539
-------------------------------------------------------------
Solution Validates: avg error less than 1.000000e-13 on all three arrays
-------------------------------------------------------------
```
