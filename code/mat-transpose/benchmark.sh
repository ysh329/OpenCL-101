#!/bin/bash

max_iter=8

kernel_arr=("matrixTranspose_v1.cl" "matrixTranspose_v2.cl")


for kernel in ${kernel_arr[@]}
do
	size=16
	for ((run_idx = 1; run_idx <= $max_iter ; run_idx++))
	do
		size=$[2*$size]
		printf "RUNING %s x %s\n" $size $size
		./matrixTranspose $size $size $kernel
	done
done

