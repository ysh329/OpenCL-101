#!/bin/python

def init():
    global bandwidth_bin
    global size_list
    global kernel_file_path
    global kernel_func_list
    global run_num
    global global_work_size_list

    bandwidth_bin = "./bandwidth"
    size_list = map(lambda n: 2**n, xrange(4, 12))
    kernel_file_path = "bandwidth_kernel.cl"
    kernel_func_list = ["global_bandwidth_float_v1"]
    run_num = 100
    global_work_size_list = []


def create_cmd():
    global cmd_list
    cmd_list = []
    import re
    vec_degree_pattern = "global_bandwidth.*_v(.*)"

    for func_idx in xrange(len(kernel_func_list)):
        kernel_func = kernel_func_list[func_idx]
        for size_idx in xrange(len(size_list)):
            size = size_list[size_idx]
            vec_degree = int(re.findall(vec_degree_pattern, kernel_func)[0])
            global_work_size = [(size/vec_degree)**2, 1, 1]
            # create cmd
            tmp_cmd = [bandwidth_bin, size, size, kernel_file_path, kernel_func, run_num, global_work_size[0], global_work_size[1], global_work_size[2]]
            cmd = " ".join(map(str, tmp_cmd))
            cmd_list.append(cmd)
            
    # check
    for cmd in cmd_list: print(cmd)

def run_cmd():
    import os
    for cmd_idx in xrange(len(cmd_list)):
        os.system(cmd)

if __name__ == "__main__":
    init()
    create_cmd()
    #run_cmd()
