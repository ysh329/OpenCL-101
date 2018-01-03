#!/bin/python

DEBUG = False

def init(range_start_num, range_max_num):
    range_list = range(range_start_num, range_max_num)
    range_list.extend([1])
    range_list.sort()

    return range_list



def calc_valid_lws(gws_tuple, range_list):
    # init `gws_dict` dict, storing corresponding set of `local work size`
    gws_dict = dict()

    for gws_idx in xrange(len(gws_tuple)):
        gws_num = gws_tuple[gws_idx]
        lws_set = set()

        for ws_idx in xrange(len(range_list)):
            work_size = range_list[ws_idx]
            if gws_num % work_size == 0:
                lws_set.add(work_size)

        gws_dict[gws_num] = lws_set
        
    if DEBUG:
        for k in gws_dict:
            print(k, gws_dict[k])


    lws_tuple_list = []
    for lws0_num in gws_dict[gws_tuple[0]]:
        for lws1_num in gws_dict[gws_tuple[1]]:
            lws_tuple = (lws0_num, lws1_num)
            lws_tuple_list.append(lws_tuple)

    if DEBUG:
        for idx in xrange(len(lws_tuple_list)):
            print(idx, lws_tuple_list[idx])

    return lws_tuple_list


def gen_cmd(lws_tuple_list):
    cmd_list = []
    for lws_tuple in lws_tuple_list:
        cmd = \
    """./matMultWithInterleaveTrans 1024 1024 1020 \
./gemm_interleave_trans.c mat_trans_vec4 1020 256 1 \
./gemm_interleave_trans.c mat_interleave_vec4  256 255 1 \
./gemm_interleave_trans.c gemm_interleaved_transposed_vec4 256 256 1   {0} {1} 1 \
1 100 """.format(lws_tuple[0], lws_tuple[1])
#        cmd = \
#    """./matMultWithInterleaveTrans 2048 2048 2040 \
#./gemm_interleave_trans.c mat_trans_vec4 2040 512 1 \
#./gemm_interleave_trans.c mat_interleave_vec4 512 510 1 \
#./gemm_interleave_trans.c gemm_interleaved_transposed_vec4 512 512 1    {0} {1} 1 \
#1 100""".format(lws_tuple[0], lws_tuple[1])
        if DEBUG:
            print(cmd)
        cmd_list.append(cmd)

    if DEBUG:
        for idx in range(len(cmd_list)):
            print(idx), 
            print(cmd_list[idx])

    return cmd_list



def run_cmd(cmd_list):
    from os import system
    for cmd_idx in xrange(len(cmd_list)):
        cmd = cmd_list[cmd_idx]
        if DEBUG:
            print cmd
            system(cmd)
            break
        system(cmd)


if __name__ == "__main__":
    # init params
    range_start_num = 4
    range_max_num = 256
    # max dimension of `gws_tuple` is 2
    gws_tuple = (1024, 1024)

    range_list = init(range_start_num=range_start_num,
                      range_max_num=range_max_num)

    lws_tuple_list = calc_valid_lws(gws_tuple=gws_tuple,
                                    range_list=range_list)

    cmd_list = gen_cmd(lws_tuple_list)
    run_cmd(cmd_list)

