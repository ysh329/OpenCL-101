#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// float 1024x1024x1024 0.703824 s 3.051168 GFLOPS
// half  1024x1024x1024 0.606415 s 3.541280 GFLOPS 
__kernel void mat_mult_naive(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p++) {
        res += a[row * K + p] * b[p * N + col];
    }
    c[row * N + col] = res;
}


__kernel void pp(__global const CL_INPUT_TYPE *bb,
                 __global CL_INPUT_TYPE *cc) {
    cc[0] = cc[0] * 1;
}



__kernel void pack_blocked_mat_b(const int k, 
                                 __global CL_INPUT_TYPE *b, 
                                 const int ldb, 
                                 __global CL_INPUT_TYPE *b_to) {
   /*
        b: <kc, 4>
        b_to: <1, kc*4>
    _   |--->  4  <---|
    |   bi0 bi1 bi2 bi3 
    |   bi0 bi1 bi2 bi3 
    kc  bi0 bi1 bi2 bi3 
    |   ... ... ... ... 
    |   ... ... ... ...
    |   ... ... ... ...
    _   ... ... ... ... 
    */
    for (int i = 0; i < k; i++) {
        __global CL_INPUT_TYPE
            *b_ij_pntr = b+i*ldb;

        *(b_to+0) = *(b_ij_pntr+0);
        *(b_to+1) = *(b_ij_pntr+1);
        *(b_to+2) = *(b_ij_pntr+2);
        *(b_to+3) = *(b_ij_pntr+3);

        b_to += 4;
    }
}

__kernel void pack_blocked_mat_a(const int k, 
                                 __global CL_INPUT_TYPE *a, 
                                 const int lda, 
                                 __global CL_INPUT_TYPE *a_to) {
    /*
         a: <4, kc>
         a_to: <1, kc*4>
     _   |----->  kc  <-----|
     |   a0j ... ... ... ...
     4   a1j ... ... ... ...
     |   a2j ... ... ... ...
     |   a3j ... ... ... ...
     ^
     */

    __global CL_INPUT_TYPE
        *a0j_pntr = a+0*lda+0,
        *a1j_pntr = a+1*lda+0,
        *a2j_pntr = a+2*lda+0,
        *a3j_pntr = a+3*lda+0;

    for (int j = 0; j < k; j++) {
        *a_to++ = *a0j_pntr++;
        *a_to++ = *a1j_pntr++;
        *a_to++ = *a2j_pntr++;
        *a_to++ = *a3j_pntr++;
    }
}


#define mc 256 
#define kc 128
#define nb 1000

__kernel void mat_mult_naive_blocked(const int M, const int N, const int K, 
                                     __global const CL_INPUT_TYPE *a, const int lda,
                                     __global const CL_INPUT_TYPE *b, const int ldb,
                                     __global       CL_INPUT_TYPE *c, const int ldc,
                                     const int first_time) 
{
    const int i = get_global_id(0) * 4; // row
    const int j = get_global_id(1) * 4; // col

    __global CL_INPUT_TYPE packedA[ M * N ];
    __global CL_INPUT_TYPE packedB[ kc*nb ];

    if (first_time) 
        pack_blocked_mat_b(K, &b[j*ldb+0], ldb, &packedB[j*K]);
    if (j == 0)
        pack_blocked_mat_a(K, &a[i*lda+0], lda, &packedA[i*K]);

    AddDot4x4( k, &packedA[ i*k ], 4, &packedB[ j*K ], K, &c( i,j ), ldc );
}

__kernel void AddDot4x4(const int k, 
                        __global CL_INPUT_TYPE *a, const int lda,
                        __global CL_INPUT_TYPE *b, const int ldb,
                        __global CL_INPUT_TYPE *c, const int ldc) {
    int p;
    CL_ELEM_TYPE
        c00, c01, c02, c03,
        c10, c11, c12, c13,
        c20, c21, c22, c23,
        c30, c31, c32, c33,
        bp0, bp1, bp2, bp3,
        a0p, a1p, a2p, a3p;

    c00=0; c01=0; c02=0; c03=0;
    c10=0; c11=0; c12=0; c13=0;
    c20=0; c21=0; c22=0; c23=0;
    c30=0; c31=0; c32=0; c33=0;

    for (int p = 0; p < k; p++) {
        a0p = *(a+0);
        a1p = *(a+1);
        a2p = *(a+2);
        a3p = *(a+3);
        a += 4;

        bp0 = *(b+0);
        bp1 = *(b+1);
        bp2 = *(b+2);
        bp3 = *(b+3);
        b += 4;

        c00+=a0p*bp0;  c01+=a0p*bp1;  c02+=a0p*bp2;  c03+=a0p*bp3;
        c10+=a1p*bp0;  c11+=a1p*bp1;  c12+=a1p*bp2;  c13+=a1p*bp3;
        c20+=a2p*bp0;  c21+=a2p*bp1;  c22+=a2p*bp2;  c23+=a2p*bp3;
        c30+=a3p*bp0;  c31+=a3p*bp1;  c32+=a3p*bp2;  c13+=a3p*bp3;
    }

    c[0*ldc+0]+=c00; c[0*ldc+1]+=c01; c[0*ldc+2]+=c02; c[0*ldc+3]+=c03;
    c[1*ldc+0]+=c10; c[1*ldc+1]+=c11; c[1*ldc+2]+=c12; c[1*ldc+3]+=c13;
    c[2*ldc+0]+=c20; c[2*ldc+1]+=c21; c[2*ldc+2]+=c22; c[2*ldc+3]+=c23;
    c[3*ldc+0]+=c30; c[3*ldc+1]+=c31; c[3*ldc+2]+=c32; c[3*ldc+3]+=c33;

}

__kernel void mat_mult_naive_trans(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *bT, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p++) {
        bT[col * K + p] = b[N * p + col];
        res += a[row * M + p] * bT[col * K + p];
    }
    c[row * N + col] = res;
}

/*
__kernel void mat_mult_naive4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;
    CL_ELEM_TYPE aa, bb;

    for (int p = 0; p < K; p += 4) {

        // 2 row elems: a[row * M + p], a[row * M + p + 1]
        aa = *((__global CL_ELEM_TYPE *)(a + row * M + p));

        // 2 col elems: b[p * N + col], b[(p+1) * N + col]
        bb = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+2) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+3) * N + col),
                    );
        res += aa * bb;
    }
    c[row * N + col] = res.s0 + res.s1 + res.s2 + res.s3;
}
*/
