#!/bin/bash
file_name=$0
cur_dir="$(pwd)""${file_name:1}"
echo $cur_dir
cat $cur_dir
gcc -o run_performance performance.c -std=gnu99 -O3 -I/opt/AMDAPPSDK-3.0/include/ -lOpenCL -lm
