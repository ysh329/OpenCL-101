#!/bin/python2

def init():
	global matrix_transpose_bin, ndim_list, size_list, run_num, kernel_file_path_list, global_work_size_2d_list

	############################# initialize parameters #############################
	matrix_transpose_bin = "./matrixTranspose"
	#kernel_file_path_list = ["./matrixTranspose_v1.cl", "./matrixTranspose_v2.cl"]
	kernel_file_path_list = ["./matrixTranspose_v1_4x1.cl"]
	run_num = 100
	size_list = map(lambda n: 2**n, xrange(4,14))
	#################################################################################

	global_work_size_2d_list = []
	#global_work_size_list_4_kernel_1 = map(lambda size: (size,    1, 1), size_list)
	global_work_size_list_4_kernel_1_1x4 = map(lambda size: (size/4,    1, 1), size_list)
	#global_work_size_list_4_kernel_2 = map(lambda size: (size, size, 1), size_list)
	#global_work_size_2d_list.append(global_work_size_list_4_kernel_1)
	#global_work_size_2d_list.append(global_work_size_list_4_kernel_2)

	global_work_size_2d_list.append(global_work_size_list_4_kernel_1_1x4)

	#################################################################################

	# check value
	#print(size_list)
	#print(kernel_file_path_list)
	#for i in global_work_size_2d_list: print i

def create_cmd():

	global cmd_list
	cmd_list = []

	for kernel_idx in xrange(len(kernel_file_path_list)):
		global_work_size_list = global_work_size_2d_list[kernel_idx]
		kernel_file_path = kernel_file_path_list[kernel_idx]
		for size_idx in xrange(len(size_list)):
			global_work_size = global_work_size_list[size_idx]
			size = size_list[size_idx]
			# concate cmd
			tmp_list = [matrix_transpose_bin, size, size, kernel_file_path, run_num, global_work_size[0], global_work_size[1], global_work_size[2]]
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
