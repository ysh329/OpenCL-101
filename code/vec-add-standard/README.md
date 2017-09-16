# vec-add-standard


## Build

Compilation command:

```shell
$ ./make.sh
```
## Usage

Use following command to execute vector addition of OpenCL:

```shell
$ ./vec-add-standard
```

It'll output messages below:
```shell
$ ./vec-add-standard 

========== INIT VALUE ==========
a: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
b: 0 1 0 0 1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 0 0 1 1 0 
c: 1 2 1 1 2 1 2 1 1 1 2 2 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 1 2 2 1 2 1 1 2 2 1 1 2 1 1 1 2 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 2 1 1 2 2 1 

========== CHECK RESULT cpu-verison && gpu-version ==========
c_d: 1 2 1 1 2 1 2 1 1 1 2 2 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 1 2 2 1 2 1 1 2 2 1 1 2 1 1 1 2 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 2 1 1 2 2 1 
correct rate: 64 / 64 , 1.00
len-1=63, c_d[63]==c[63]: 1, c_d[63]=1, c[63]=1 

========== CHECK RESULT ELEMENT BY ELEMENT ==========
idx  c  c_d
 0  1  1 
 1  2  2 
 2  1  1 
 3  1  1 
 4  2  2 
 5  1  1 
 6  2  2 
 7  1  1 
 8  1  1 
 9  1  1 
10  2  2 
11  2  2 
12  2  2 
13  1  1 
14  2  2 
15  1  1 
16  2  2 
17  2  2 
18  1  1 
19  1  1 
20  2  2 
21  2  2 
22  2  2 
23  2  2 
24  2  2 
25  1  1 
26  2  2 
27  1  1 
28  2  2 
29  2  2 
30  1  1 
31  2  2 
32  1  1 
33  1  1 
34  2  2 
35  2  2 
36  1  1 
37  1  1 
38  2  2 
39  1  1 
40  1  1 
41  1  1 
42  2  2 
43  1  1 
44  1  1 
45  1  1 
46  1  1 
47  2  2 
48  2  2 
49  1  1 
50  1  1 
51  1  1 
52  1  1 
53  2  2 
54  2  2 
55  2  2 
56  2  2 
57  2  2 
58  2  2 
59  1  1 
60  1  1 
61  2  2 
62  2  2 
63  1  1 
```


## Appendix

### Enable `-lm` compilation choice

Because of new verison GCC, which spilt standard library in C99 into two parts (`libc`, `libm`), some math library like `<math.h>` in `libm` need add compilation choice `-lm` (that's link libm) when using, otherwise it will error.

由于在gcc新的版本中GCC把C99中的标准库分成了libc和libm两个部分，libm中包括一些数学库<math.h>等（我这里使用了ceil()这个函数），所以如果要使用libm时则必须加上编译选项-lm(即link libm)，不然会报错。

### reference
* [undefined reference to ceil 链接错误 . - 上善若水的日志 - 网易博客](http://blog.163.com/zsy_19880518/blog/static/18525812720130631537226/)
* [undefined reference to symbol xx@@GLIBC_2.2.5 - 你越努力，你的运气就会越好 - CSDN博客](http://blog.csdn.net/vintionnee/article/details/36004973)

