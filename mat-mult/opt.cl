#pragma OPENCL EXTENSION cl_khr_fp16 : enable

////////////////////////////////////////////////////////////////
//// Optimizaton0: naive implementation
////////////////////////////////////////////////////////////////
// float 1024x1024x1024 0.738184 s 2.909143 GFLOPS

////////////////////////////////////////////////////////////////

__kernel void mat_mult_naive(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;
    for (int p = 0; p < K; p++) {
        res += a[row * K + p] * b[p * N + col];
    }
    c[row * N + col] = res;
}


////////////////////////////////////////////////////////////////
//// Optimizaton_1
////////////////////////////////////////////////////////////////
// float 1024x1024x1024 0.936172 s 2.293900 GFLOPS

////////////////////////////////////////////////////////////////
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

__kernel void AddDot1(const int K, __global const CL_INPUT_TYPE *x, const int N, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma);

__kernel void mat_mult_1(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    AddDot1(K, &a[K * row + 0], N, &b[N * 0 + col], &c[N * row + col]);
}

__kernel void AddDot1(const int K, __global const CL_INPUT_TYPE *x, const int N, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma) {
    *gamma = 0;
    for (int p = 0; p < K; p++) {
        *gamma += x[p] * y[p*N];
    }
}


////////////////////////////////////////////////////////////////
//// Optimizaton_2
////////////////////////////////////////////////////////////////
// float 1024x1024x1024 2.271603 s 0.945361 GFLOPS 
//
///////////////////////////////////

__kernel void AddDot2(const int K, __global const CL_INPUT_TYPE *x, const int N, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma);

__kernel void mat_mult_2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);

    AddDot2(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);
    AddDot2(K, &a[K * row + 0], N, &b[N * 0 + (col+1)], &c[N * row + (col+1)]);
    AddDot2(K, &a[K * row + 0], N, &b[N * 0 + (col+2)], &c[N * row + (col+2)]);
    AddDot2(K, &a[K * row + 0], N, &b[N * 0 + (col+3)], &c[N * row + (col+3)]);

}

__kernel void AddDot2(const int K, __global const CL_INPUT_TYPE *x, const int N, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma) {
    *gamma = 0;   
    for (int p = 0; p < K; p++) {
        *gamma += x[p] * y[p*N];
    }
}

////////////////////////////////////////////////////////////////
//// Optimizaton_3_1x4
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 11.846945 s 0.181269 GFLOPS
////////////////////////////////////////////////////////////////
__kernel void AddDot_3_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);
__kernel void AddDot3(const int K, __global const CL_INPUT_TYPE *x, const int N, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma);

__kernel void mat_mult_3_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);

    AddDot_3_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_3_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    AddDot3(K, &aa[0], N, &bb[0+0], &cc[0+0]);
    AddDot3(K, &aa[0], N, &bb[0+1], &cc[0+1]);
    AddDot3(K, &aa[0], N, &bb[0+2], &cc[0+2]);
    AddDot3(K, &aa[0], N, &bb[0+3], &cc[0+3]);
}

__kernel void AddDot3(const int K, __global const CL_INPUT_TYPE *x, const int N, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma) {
    *gamma = 0;   
    for (int p = 0; p < K; p++) {
        *gamma += x[p] * y[p*N];
    }
}

/////////////////////////////////////////////////////////////////
//// float Strange error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//// correct rate: 0.25
/////////////////////////////////////////////////////////////////
__kernel void AddDot_3_4x1(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_3_4x1(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1) << 2;

    AddDot_3_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_3_4x1(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    AddDot3(K, &aa[0*K], N, &bb[0], &cc[0*N]);
    AddDot3(K, &aa[1*K], N, &bb[0], &cc[1*N]);
    AddDot3(K, &aa[2*K], N, &bb[0], &cc[2*N]);
    AddDot3(K, &aa[3*K], N, &bb[0], &cc[3*N]);
}

////////////////////////////////////////////////////////////////
//// Optimization 4
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 11.938108 s 0.179885 GFLOPS
////////////////////////////////////////////////////////////////

__kernel void AddDot_4_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_4_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_4_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_4_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[0] = 0;
    for (int p = 0; p < K; p++) {
        cc[0] += aa[p] * bb[p*N];
    }
    cc[1] = 0;
    for (int p = 0; p < K; p++) {
        cc[1] += aa[p] * bb[p*N+1];
    }
    cc[2] = 0;
    for (int p = 0; p < K; p++) {
        cc[2] += aa[p] * bb[p*N+2];
    }
    cc[3] = 0;
    for (int p = 0; p < K; p++) {
        cc[3] += aa[p] * bb[p*N+3];
    }

}

////////////////////////////////////////////////////////////////
//// correct rate: 0
////////////////////////////////////////////////////////////////

__kernel void AddDot_4_4x1(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_4_4x1(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1) << 2;
    
    AddDot_4_4x1(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_4_4x1(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[N*0] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*0] += aa[K*0+p] * bb[p];
    }
    cc[N*1] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*1] += aa[K*1+p] * bb[p];
    }
    cc[N*2] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*2] += aa[K*2+p] * bb[p];
    }
    cc[N*3] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*3] += aa[K*3+p] * bb[p];
    }

}



////////////////////////////////////////////////////////////////
//// Optimization 5
////////////////////////////////////////////////////////////////
//// float
////////////////////////////////////////////////////////////////









