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

