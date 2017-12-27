#include <stdio.h>

double get_gflops(double sum_duration, int run_num, int size) {
    double gflops = 2.0 * size / (double)sum_durationsize / (double)run_num * 1.0e-9;
    return gflops;
}

double get_gbps(double sum_duration, int run_num, int size, int elem_size) {
    double gbps = 2.0 * size * elem_size / (double)(1024 * 1024 * 1024) / sum_duration / (double)run_num;
    return gbps;
}


