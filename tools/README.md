# tools

This directory contains some tools, such as adjusting/outputing CPU/GPU frequency.

## How to Set CPU/GPU Frequency

### Check Frequency

So as to ensure afterwards modification is valid, you can check current frequency first as below:

```shell
# check cpu frequency
$ ./ck-print-cpu-freq
```

After executed, my machine outputs messages about **CPU freq.** as below:

```shell
$ ./ck-print-cpu-freq 
*** Current CPU frequency from scaling:
1416000
1416000
1416000
1416000
1800000
1800000
*** Min CPU frequency:
408000
408000
408000
408000
408000
408000
*** Max CPU frequency:
1416000
1416000
1416000
1416000
1800000
1800000
```

```shell
# check gpu frequency
$ ./ck-print-gpu-freq
```

Similar as **CPU frequency**, my machine outputs **GPU freq.** as below:

```shell
$ ./ck-print-gpu-freq 
*** Current GPU frequency:
800000000
*** Min frequency:
200000000
*** Max frequency:
800000000
*** Available GPU frequencies:
200000000 297000000 400000000 500000000 594000000 800000000
*** Current GPU governor:
performance
*** Available GPU governor:
userspace powersave performance simple_ondemand
```

### Set Max Frequency

Before execution, ensure you have authority to modify it or execute these script with `sudo` prefix.

```shell
# for CPU
$ sudo ./ck-set-cpu-performance

# for GPU
$ sudo ./ck-set-gpu-performance
```


