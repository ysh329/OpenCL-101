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
//// float 1024x1024x1024 2.483253 s 0.864786 GFLOPS 
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
        cc[N*0] += aa[K*0+p] * bb[p*N];
    }
    cc[N*1] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*1] += aa[K*1+p] * bb[p*N];
    }
    cc[N*2] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*2] += aa[K*2+p] * bb[p*N];
    }
    cc[N*3] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*3] += aa[K*3+p] * bb[p*N];
    }

}



////////////////////////////////////////////////////////////////
//// Optimization 5
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 1.427757 s 1.504096 GFLOPS
////////////////////////////////////////////////////////////////

__kernel void AddDot_5_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_5_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_5_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_5_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[0] = cc[1] = cc[2] = cc[3] = 0;
    for (int p = 0; p < K; p++) {
        cc[0] += aa[p] * bb[p*N];
        cc[1] += aa[p] * bb[p*N+1];
        cc[2] += aa[p] * bb[p*N+2];
        cc[3] += aa[p] * bb[p*N+3];
    }

}

////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 2.249839 s 0.954506 GFLOPS 
////////////////////////////////////////////////////////////////

__kernel void AddDot_5_4x1(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_5_4x1(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1) << 2;
    
    AddDot_5_4x1(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_5_4x1(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[N*0] = cc[N*1] = cc[N*2] = cc[N*3] = 0;
    for (int p = 0; p < K; p++) {
        cc[N*0] += aa[K*0+p] * bb[p*N];
        cc[N*1] += aa[K*1+p] * bb[p*N];
        cc[N*2] += aa[K*2+p] * bb[p*N];
        cc[N*3] += aa[K*3+p] * bb[p*N];
    }

}

////////////////////////////////////////////////////////////////
//// Optimization 6
////////////////////////////////////////////////////////////////
//// OpenCL does not support the 'register' storage class specifier
////////////////////////////////////////////////////////////////

__kernel void AddDot_6_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_6_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_6_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_6_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    //register 
      CL_INPUT_TYPE 
        cc_00_reg = 0.0,
        cc_01_reg = 0.0,
        cc_02_reg = 0.0,
        cc_03_reg = 0.0,
        aa_0p_reg = 0.0;

    for (int p = 0; p < K; p++) {
        aa_0p_reg = aa[p];
        cc_00_reg += aa_0p_reg * bb[p*N];
        cc_01_reg += aa_0p_reg * bb[p*N+1];
        cc_02_reg += aa_0p_reg * bb[p*N+2];
        cc_03_reg += aa_0p_reg * bb[p*N+3];
    }
    cc[0] = cc_00_reg;
    cc[1] = cc_01_reg;
    cc[2] = cc_02_reg;
    cc[3] = cc_03_reg;

}


////////////////////////////////////////////////////////////////
//// Optimization 7
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 1.396675 s 1.537568 GFLOPS
////////////////////////////////////////////////////////////////

__kernel void AddDot_7_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_7_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_7_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_7_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {

    __global const CL_INPUT_TYPE 
                  *bp0_pntr = &bb[0*N+0],
                  *bp1_pntr = &bb[0*N+1], 
                  *bp2_pntr = &bb[0*N+2], 
                  *bp3_pntr = &bb[0*N+3],
                  *ap0_pntr;

    cc[0] = cc[1] = cc[2] = cc[3] = 0;
    for (int p = 0; p < K; p++) {
        ap0_pntr = &aa[p];
        cc[0] += *ap0_pntr * *(bp0_pntr+p*N);
        cc[1] += *ap0_pntr * *(bp1_pntr+p*N);
        cc[2] += *ap0_pntr * *(bp2_pntr+p*N);
        cc[3] += *ap0_pntr * *(bp3_pntr+p*N);
    }

}

///////////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 1.394532 s 1.539932 GFLOPS
////////////////////////////////////////////////////////////////

__kernel void AddDot_72_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_72_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_72_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_72_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[0] = cc[1] = cc[2] = cc[3] = 0;
    __global CL_INPUT_TYPE 
                  *bp0_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+0],
                  *bp1_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+1], 
                  *bp2_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+2], 
                  *bp3_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+3],
                  *ap0_pntr,
                  *cp0_pntr = (__global CL_INPUT_TYPE *)&cc[0],
                  *cp1_pntr = (__global CL_INPUT_TYPE *)&cc[1],
                  *cp2_pntr = (__global CL_INPUT_TYPE *)&cc[2],
                  *cp3_pntr = (__global CL_INPUT_TYPE *)&cc[3];


    for (int p = 0; p < K; p++) {
        ap0_pntr = (__global CL_INPUT_TYPE *)&aa[p];
        *cp0_pntr += *ap0_pntr * *(bp0_pntr+p*N);
        *cp1_pntr += *ap0_pntr * *(bp1_pntr+p*N);
        *cp2_pntr += *ap0_pntr * *(bp2_pntr+p*N);
        *cp3_pntr += *ap0_pntr * *(bp3_pntr+p*N);
    }
}

////////////////////////////////////////////////////////////////
//// Optimization 8
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 1.435723 s 1.495751 GFLOP 
////////////////////////////////////////////////////////////////

__kernel void AddDot_8_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_8_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_8_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_8_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[0] = cc[1] = cc[2] = cc[3] = 0;
    __global CL_INPUT_TYPE 
                  *bp0_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+0],
                  *bp1_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+1], 
                  *bp2_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+2], 
                  *bp3_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+3],
                  *ap0_pntr,
                  *ap1_pntr,
                  *ap2_pntr,
                  *ap3_pntr,
                  *cp0_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+0], *cp1_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+1], *cp2_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+2], *cp3_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+3];

                  //*cp00_pntr = &cc[0*N+0], *cp01_pntr = &cc[0*N+1], *cp02_pntr = &cc[0*N+2], *cp03_pntr = &cc[0*N+3],
                  //*cp10_pntr = &cc[1*N+0], *cp11_pntr = &cc[1*N+1], *cp12_pntr = &cc[1*N+2], *cp13_pntr = &cc[1*N+3],
                  //*cp20_pntr = &cc[2*N+0], *cp21_pntr = &cc[2*N+1], *cp22_pntr = &cc[2*N+2], *cp23_pntr = &cc[2*N+3],
                  //*cp30_pntr = &cc[3*N+0], *cp31_pntr = &cc[3*N+1], *cp32_pntr = &cc[3*N+2], *cp33_pntr = &cc[3*N+3];


    for (int p = 0; p < K; p+=4) {
        ap0_pntr = (__global CL_INPUT_TYPE *)&aa[p];
        *cp0_pntr += *ap0_pntr * *(bp0_pntr+p*N);
        *cp1_pntr += *ap0_pntr * *(bp1_pntr+p*N);
        *cp2_pntr += *ap0_pntr * *(bp2_pntr+p*N);
        *cp3_pntr += *ap0_pntr * *(bp3_pntr+p*N);

        ap1_pntr = (__global CL_INPUT_TYPE *)&aa[p+1];
        *cp0_pntr += *ap1_pntr * *(bp0_pntr+(p+1)*N);
        *cp1_pntr += *ap1_pntr * *(bp1_pntr+(p+1)*N);
        *cp2_pntr += *ap1_pntr * *(bp2_pntr+(p+1)*N);
        *cp3_pntr += *ap1_pntr * *(bp3_pntr+(p+1)*N);
      
        ap2_pntr = (__global CL_INPUT_TYPE *)&aa[p+2];
        *cp0_pntr += *ap2_pntr * *(bp0_pntr+(p+2)*N);
        *cp1_pntr += *ap2_pntr * *(bp1_pntr+(p+2)*N);
        *cp2_pntr += *ap2_pntr * *(bp2_pntr+(p+2)*N);
        *cp3_pntr += *ap2_pntr * *(bp3_pntr+(p+2)*N);

        ap3_pntr = (__global CL_INPUT_TYPE *)&aa[p+3];
        *cp0_pntr += *ap3_pntr * *(bp0_pntr+(p+3)*N);
        *cp1_pntr += *ap3_pntr * *(bp1_pntr+(p+3)*N);
        *cp2_pntr += *ap3_pntr * *(bp2_pntr+(p+3)*N);
        *cp3_pntr += *ap3_pntr * *(bp3_pntr+(p+3)*N);


    }
}

////////////////////////////////////////////////////////////////
//// Optimization 9: nothing to do with opt8
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 
////////////////////////////////////////////////////////////////

__kernel void AddDot_9_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc);

__kernel void mat_mult_9_1x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);
    
    AddDot_9_1x4(K, &a[K * row + 0], N, &b[N * 0 + (col+0)], &c[N * row + (col+0)]);

}

__kernel void AddDot_9_1x4(const int K, __global const CL_INPUT_TYPE *aa, const int N, __global const CL_INPUT_TYPE *bb, __global CL_INPUT_TYPE *cc) {
    cc[0] = cc[1] = cc[2] = cc[3] = 0;
    __global CL_INPUT_TYPE 
                  *bp0_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+0],
                  *bp1_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+1], 
                  *bp2_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+2], 
                  *bp3_pntr = (__global CL_INPUT_TYPE *)&bb[0*N+3],
                  *ap0_pntr,
                  *ap1_pntr,
                  *ap2_pntr,
                  *ap3_pntr,
                  *cp0_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+0], *cp1_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+1], *cp2_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+2], *cp3_pntr = (__global CL_INPUT_TYPE *)&cc[0*N+3];

    for (int p = 0; p < K; p+=4) {
        ap0_pntr = (__global CL_INPUT_TYPE *)&aa[p];
        *cp0_pntr += *ap0_pntr * *(bp0_pntr+p*N);
        *cp1_pntr += *ap0_pntr * *(bp1_pntr+p*N);
        *cp2_pntr += *ap0_pntr * *(bp2_pntr+p*N);
        *cp3_pntr += *ap0_pntr * *(bp3_pntr+p*N);

        ap1_pntr = (__global CL_INPUT_TYPE *)&aa[p+1];
        *cp0_pntr += *ap1_pntr * *(bp0_pntr+(p+1)*N);
        *cp1_pntr += *ap1_pntr * *(bp1_pntr+(p+1)*N);
        *cp2_pntr += *ap1_pntr * *(bp2_pntr+(p+1)*N);
        *cp3_pntr += *ap1_pntr * *(bp3_pntr+(p+1)*N);
      
        ap2_pntr = (__global CL_INPUT_TYPE *)&aa[p+2];
        *cp0_pntr += *ap2_pntr * *(bp0_pntr+(p+2)*N);
        *cp1_pntr += *ap2_pntr * *(bp1_pntr+(p+2)*N);
        *cp2_pntr += *ap2_pntr * *(bp2_pntr+(p+2)*N);
        *cp3_pntr += *ap2_pntr * *(bp3_pntr+(p+2)*N);

        ap3_pntr = (__global CL_INPUT_TYPE *)&aa[p+3];
        *cp0_pntr += *ap3_pntr * *(bp0_pntr+(p+3)*N);
        *cp1_pntr += *ap3_pntr * *(bp1_pntr+(p+3)*N);
        *cp2_pntr += *ap3_pntr * *(bp2_pntr+(p+3)*N);
        *cp3_pntr += *ap3_pntr * *(bp3_pntr+(p+3)*N);
    }
}


////////////////////////////////////////////////////////////////
//// Optimization 10_3
////////////////////////////////////////////////////////////////
//// float 1024x1024x1024 low efficient verison
//// 1024x1024x1024 2.709294 s 0.792636 GFLOPS 
////////////////////////////////////////////////////////////////

__kernel void mat_mult_10_3_4x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1) << 2;

    //*(c+row*N+col)     = 0;  *(c+row*N+(col+1))     = 0;  *(c+row*N+(col+2))     = 0;  *(c+row*N+(col+3))     = 0;
    //*(c+(row+1)*N+col) = 0;  *(c+(row+1)*N+(col+1)) = 0;  *(c+(row+1)*N+(col+2)) = 0;  *(c+(row+1)*N+(col+3)) = 0;
    //*(c+(row+2)*N+col) = 0;  *(c+(row+2)*N+(col+1)) = 0;  *(c+(row+2)*N+(col+2)) = 0;  *(c+(row+2)*N+(col+3)) = 0;
    //*(c+(row+3)*N+col) = 0;  *(c+(row+3)*N+(col+1)) = 0;  *(c+(row+3)*N+(col+2)) = 0;  *(c+(row+3)*N+(col+3)) = 0;

    

    for (int p = 0; p < K; p++) {
        // 1st row of c
        *(c+row*N+col)         += *(a+row*K+p)     * *(b+p*N+col);
        *(c+row*N+(col+1))     += *(a+row*K+p)     * *(b+p*N+(col+1));
        *(c+row*N+(col+2))     += *(a+row*K+p)     * *(b+p*N+(col+2));
        *(c+row*N+(col+3))     += *(a+row*K+p)     * *(b+p*N+(col+3));
        // 2nd row of c
        *(c+(row+1)*N+col)     += *(a+(row+1)*K+p) * *(b+p*N+col);
        *(c+(row+1)*N+(col+1)) += *(a+(row+1)*K+p) * *(b+p*N+(col+1));
        *(c+(row+1)*N+(col+2)) += *(a+(row+1)*K+p) * *(b+p*N+(col+2));
        *(c+(row+1)*N+(col+3)) += *(a+(row+1)*K+p) * *(b+p*N+(col+3));
        // 3rd row of c
        *(c+(row+2)*N+col)     += *(a+(row+2)*K+p) * *(b+p*N+col);
        *(c+(row+2)*N+(col+1)) += *(a+(row+2)*K+p) * *(b+p*N+(col+1));
        *(c+(row+2)*N+(col+2)) += *(a+(row+2)*K+p) * *(b+p*N+(col+2));
        *(c+(row+2)*N+(col+3)) += *(a+(row+2)*K+p) * *(b+p*N+(col+3));
        // 4th row of c
        *(c+(row+3)*N+col)     += *(a+(row+3)*K+p) * *(b+p*N+col);
        *(c+(row+3)*N+(col+1)) += *(a+(row+3)*K+p) * *(b+p*N+(col+1));
        *(c+(row+3)*N+(col+2)) += *(a+(row+3)*K+p) * *(b+p*N+(col+2));
        *(c+(row+3)*N+(col+3)) += *(a+(row+3)*K+p) * *(b+p*N+(col+3));
    }
}

////////////////////////////////////////////////////////////////
////
////////////////////////////////////////////////////////////////

__kernel void mat_mult_10_4_4x4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1) << 2;

    CL_INPUT_TYPE
        c00 = 0, c01 = 0, c02 = 0, c03 = 0,
        c10 = 0, c11 = 0, c12 = 0, c13 = 0,
        c20 = 0, c21 = 0, c22 = 0, c23 = 0,
        c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    for (int p = 0; p < K; p++) {

        CL_INPUT_TYPE 
            a00 = *(a+row*K+p), 
            a10 = *(a+(row+1)*K+p), 
            a20 = *(a+(row+2)*K+p), 
            a30 = *(a+(row+3)*K+p),  

            b00 = *(b+p*N+col),  b01 = *(b+p*N+(col+1)),  b02 = *(b+p*N+(col+2)),  b03 = *(b+p*N+(col+3));
        
        c00 += a00 * b00; c01 += a00 * b01; c02 += a00 * b02; c03 += a00 * b03;
        c10 += a10 * b00; c11 += a10 * b01; c12 += a10 * b02; c13 += a10 * b03;
        c20 += a20 * b00; c21 += a20 * b01; c22 += a20 * b02; c23 += a20 * b03;
        c30 += a30 * b00; c31 += a30 * b01; c32 += a30 * b02; c33 += a30 * b03;
    }
    c[row*N+col] = c00;     c[row*N+(col+1)] = c01;     c[row*N+(col+2)] = c02;     c[row*N+(col+3)] = c03;
    c[(row+1)*N+col] = c10; c[(row+1)*N+(col+1)] = c11; c[(row+1)*N+(col+2)] = c12; c[(row+1)*N+(col+3)] = c13;
    c[(row+2)*N+col] = c20; c[(row+2)*N+(col+1)] = c21; c[(row+2)*N+(col+2)] = c22; c[(row+2)*N+(col+3)] = c23;
    c[(row+3)*N+col] = c30; c[(row+3)*N+(col+1)] = c31; c[(row+3)*N+(col+2)] = c32; c[(row+3)*N+(col+3)] = c33;
}
