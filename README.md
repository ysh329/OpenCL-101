# OpenCL-101
Learn OpenCL step by step as below.

* ./code/hello-world-from-kernel
* ./code/tutorial-1.c  
* ./code/tutorial-2.c  

## Installation guide of OpenCL 

You can choose one or two ways to use OpenCL:  
1. Install OpenCL on Ubuntu 16.04 64-bit  
2. Using OpenCL by Docker

### Install OpenCL on Ubuntu 16.04 64-bit

```Shell
# below instructions refer linux安装opencl：ubuntu14.04+opencl1.1 - qccz123456的博客 - CSDN博客
# http://blog.csdn.net/qccz123456/article/details/52606788

$ sudo apt-get update
$ sudo apt-get install build-essential g++ cmake
$ sudo apt-get install clang libclang-3.4-dev libclang-dev libclang1
$ sudo apt-get install ocl-icd-opencl-dev ocl-icd-libopencl1
$ sudo apt-get install opencl-headers ocl-icd-dev ocl-icd-libopencl1

# below instructions refer Ubuntu 16.04.2 下为 Intel 显卡启用 OpenCL_Linux教程_Linux公社-Linux系统门户网站
# http://www.linuxidc.com/Linux/2017-03/141455.htm

$ sudo apt install ocl-icd-libopencl1
$ sudo apt install opencl-headers
$ sudo apt install clinfo
$ sudo apt install ocl-icd-opencl-dev
$ sudo apt install beignet
```

### Using OpenCL by Docker

Using Docker is convenient, which you don't need config and install enviroments for all about OpenCL. Of course, [install Docker Community Edition](https://docs.docker.com/) first and then search relative images in [DockerHub](https://hub.docker.com/).

After finish Docker installation, please follow [this instruction from chihchun/opencl-intel](https://hub.docker.com/r/chihchun/opencl-intel/). If anything goes normally, using command below in command line: 

```Shell
$ docker run -t -i --device /dev/dri:/dev/dri \
chihchun/hashcat-beignet hashcat -b
```

It will print similar messages as *Verify installation*.

### Verify Installation

Using instruction below, successful installation will print same following messages:
```shell
$ clinfo

# print message below

Number of platforms                               1
  Platform Name                                   Intel Gen OCL Driver
  Platform Vendor                                 Intel
  Platform Version                                OpenCL 1.2 beignet 1.1.1
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_spir cl_khr_icd
  Platform Extensions function suffix             Intel

  Platform Name                                   Intel Gen OCL Driver
Number of devices                                 1
  Device Name                                     Intel(R) HD Graphics IvyBridge M GT2
  Device Vendor                                   Intel
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 1.2 beignet 1.1.1
  Driver Version                                  1.1.1
  Device OpenCL C Version                         OpenCL C 1.2 beignet 1.1.1
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               16
  Max clock frequency                             1000MHz
  Device Partition                                (core)
    Max number of sub-devices                     1
    Supported partition types                     None, None, None
  Max work item dimensions                        3
  Max work item sizes                             512x512x512
  Max work group size                             512
  Preferred work group size multiple              16
  Preferred / native vector sizes                 
    char                                                16 / 8       
    short                                                8 / 8       
    int                                                  4 / 4       
    long                                                 2 / 2       
    half                                                 0 / 8        (n/a)
    float                                                4 / 4       
    double                                               0 / 2        (n/a)
  Half-precision Floating-point support           (n/a)
  Single-precision Floating-point support         (core)
    Denormals                                     No
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 No
    Round to infinity                             No
    IEEE754-2008 fused multiply-add               No
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Double-precision Floating-point support         (n/a)
  Address bits                                    32, Little-Endian
  Global memory size                              2147483648 (2GiB)
  Error Correction support                        No
  Max memory allocation                           1073741824 (1024MiB)
  Unified memory for Host and Device              Yes
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       1024 bits (128 bytes)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        8192
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            65536 pixels
    Max 1D or 2D image array size                 2048 images
    Max 2D image size                             8192x8192 pixels
    Max 3D image size                             8192x8192x2048 pixels
    Max number of read image args                 128
    Max number of write image args                8
  Local memory type                               Global
  Local memory size                               65536 (64KiB)
  Max constant buffer size                        134217728 (128MiB)
  Max number of constant args                     8
  Max size of kernel argument                     1024
  Queue properties                                
    Out-of-order execution                        No
    Profiling                                     Yes
  Prefer user sync for interop                    Yes
  Profiling timer resolution                      80ns
  Execution capabilities                          
    Run OpenCL kernels                            Yes
    Run native kernels                            Yes
    SPIR versions                                 <printDeviceInfo:138: get   SPIR versions size : error -30>
  printf() buffer size                            1048576 (1024KiB)
  Built-in kernels                                __cl_copy_region_align4;__cl_copy_region_align16;__cl_cpy_region_unalign_same_offset;__cl_copy_region_unalign_dst_offset;__cl_copy_region_unalign_src_offset;__cl_copy_buffer_rect;__cl_copy_image_1d_to_1d;__cl_copy_image_2d_to_2d;__cl_copy_image_3d_to_2d;__cl_copy_image
_2d_to_3d;__cl_copy_image_3d_to_3d;__cl_copy_image_2d_to_buffer;__cl_copy_image_3d_to_buffer;__cl_copy_buffer_to_image_2d;__cl_copy_buffer_to_image_3d;__cl_fill_region_unalign;__cl_fill_region_align2;__cl_fill_region_align4;__cl_fill_region_align8_2;__cl_fill_region_align8_4;__cl_fill_region_align8_8;__cl_fill_region_
align8_16;__cl_fill_region_align128;__cl_fill_image_1d;__cl_fill_image_1d_array;__cl_fill_image_2d;__cl_fill_image_2d_array;__cl_fill_image_3d;
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_spir cl_khr_icd

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  Intel Gen OCL Driver
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [Intel]
  clCreateContext(NULL, ...) [default]            Success [Intel]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
    Platform Name                                 Intel Gen OCL Driver
    Device Name                                   Intel(R) HD Graphics IvyBridge M GT2
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  Success (1)
    Platform Name                                 Intel Gen OCL Driver
    Device Name                                   Intel(R) HD Graphics IvyBridge M GT2

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.2.8
  ICD loader Profile                              OpenCL 1.2
        NOTE:   your OpenCL library declares to support OpenCL 1.2,
                but it seems to support up to OpenCL 2.1 too.  

```


## How to compile OpenCL example in GCC?  
Precisely, the kernel compilation in OpenCL is make in running time (library call). 

In Gcc, for compilation, you only need the headers (aviables on Kronos site). But for linkage, you have to install OpenCL compatible driver.

in the Makefile :  
* for Mac OSX : -framework OpenCL 
* for Linux : -lOpenCL

ref: How to compile OpenCL example in GCC?  
https://forums.khronos.org/showthread.php/5728-How-to-compile-OpenCL-example-in-GCC

# Other problems

## git error: unable to auto-detect email address

```shell
yuanshuai@firefly:~/code/OpenCL-101$ git commit -m "update README.md"

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'yuanshuai@firefly.(none)')
```

After following instructions above, it still occured same error message. I reset `user.email` and `user.name` using `git config --local user.email "you@example.com"` and `git config --local user.name "Your name"` and it's okay!

ref: git中报unable to auto-detect email address 错误的解决拌办法 - liufangbaishi2014的博客 - CSDN博客
http://blog.csdn.net/liufangbaishi2014/article/details/50037507

