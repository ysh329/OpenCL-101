#!/bin/python2

def init():
	global matrix_transpose_bin, ndim_list, size_list, run_num, kernel_file_path_list

	matrix_transpose_bin = "./matrixTranspose"
	# ndim_list is respect to kernel_file_path_list
	ndim_list = [1, 2]
	size_list = map(lambda n: 2**n, xrange(4,14))
	kernel_max_version_num = 2
	run_num = 100

	kernel_file_path_prefix = "matrixTranspose_v"
	kernel_file_path_suffix = ".cl"

	kernel_file_path_list = map(lambda version_num:\
											 str(kernel_file_path_prefix) +\
											 str(version_num) +\
											 str(kernel_file_path_suffix),\
									 xrange(1, kernel_max_version_num+1))

	# check value
	print(size_list)
	print(kernel_file_path_list)

def create_cmd():

	global cmd_list
	cmd_list = []

	for kernel_idx in xrange(len(kernel_file_path_list)):
		ndim = ndim_list[kernel_idx]
		kernel_file_path = kernel_file_path_list[kernel_idx]
		for size_idx in xrange(len(size_list)):
			size = size_list[size_idx]
			# concate cmd
			tmp_list = [matrix_transpose_bin, ndim, size, size, kernel_file_path, run_num]
			tmp_str_list = map(str, tmp_list)
			cmd = " ".join(tmp_str_list)
			cmd_list.append(cmd)

	# check value
	for i in cmd_list: print i
			
			
	

def run_cmd():
	import os
	for cmd_idx in xrange(len(cmd_list)):
		cmd = cmd_list[cmd_idx]
		os.system(cmd)

if __name__ == "__main__":
	init()
	create_cmd()
	run_cmd()
