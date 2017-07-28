# vec-add

```shell
gcc -o vec-add vec-add.c -lOpenCL -lm
```

# ref  

## undefined reference to ceil 链接错误 . - 上善若水的日志 - 网易博客  
http://blog.163.com/zsy_19880518/blog/static/18525812720130631537226/

由于在gcc新的版本中GCC把C99中的标准库分成了libc和libm两个部分，libm中包括一些数学库<math.h>等（我这里使用了ceil()这个函数），所以如果要使用libm时则必须加上编译选项-lm(即link libm)，不然会报错。

ref:
undefined reference to symbol xx@@GLIBC_2.2.5 - 你越努力，你的运气就会越好 - CSDN博客
http://blog.csdn.net/vintionnee/article/details/36004973


