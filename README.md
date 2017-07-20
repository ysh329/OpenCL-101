# OpenCL-101
Learn OpenCL step by step.



## How to compile OpenCL example in GCC?  
Precisely, the kernel compilation in OpenCL is make in running time (library call). 

In Gcc, for compilation, you only need the headers (aviables on Kronos site). But for linkage, you have to install OpenCL compatible driver.

in the Makefile :  
* for Mac OSX : -framework OpenCL 
* for Linux : -lOpenCL

ref: How to compile OpenCL example in GCC?  
https://forums.khronos.org/showthread.php/5728-How-to-compile-OpenCL-example-in-GCC


