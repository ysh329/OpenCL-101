__kernel void matrixTranspose(const int heightA,
							                const int widthA,
							                __global const float *a,
							                __global float *a_T) {
  const int alpha = 8;
	const int colA = get_global_id(0) * alpha;
	const int rowA = get_global_id(1) * alpha;

	if ((rowA < heightA) && (colA < widthA)) {

    __global float *src_vec1x8_0 = a + (rowA+0) * widthA + (colA+0);
    __global float *src_vec1x8_1 = a + (rowA+1) * widthA + (colA+0);
    __global float *src_vec1x8_2 = a + (rowA+2) * widthA + (colA+0);
    __global float *src_vec1x8_3 = a + (rowA+3) * widthA + (colA+0);

    __global float *src_vec1x8_4 = a + (rowA+4) * widthA + (colA+0);
    __global float *src_vec1x8_5 = a + (rowA+5) * widthA + (colA+0);
    __global float *src_vec1x8_6 = a + (rowA+6) * widthA + (colA+0);
    __global float *src_vec1x8_7 = a + (rowA+7) * widthA + (colA+0);
     
    float8 vec1x8_0 = *((__global float8 *)src_vec1x8_0);
    float8 vec1x8_1 = *((__global float8 *)src_vec1x8_1);
    float8 vec1x8_2 = *((__global float8 *)src_vec1x8_2);
    float8 vec1x8_3 = *((__global float8 *)src_vec1x8_3);

    float8 vec1x8_4 = *((__global float8 *)src_vec1x8_4);
    float8 vec1x8_5 = *((__global float8 *)src_vec1x8_5);
    float8 vec1x8_6 = *((__global float8 *)src_vec1x8_6);
    float8 vec1x8_7 = *((__global float8 *)src_vec1x8_7);

		a_T[(colA+0) * heightA + (rowA+0)] = vec1x8_0.s0;
		a_T[(colA+1) * heightA + (rowA+0)] = vec1x8_0.s1;
   	a_T[(colA+2) * heightA + (rowA+0)] = vec1x8_0.s2;
    a_T[(colA+3) * heightA + (rowA+0)] = vec1x8_0.s3;
    a_T[(colA+4) * heightA + (rowA+0)] = vec1x8_0.s4;
    a_T[(colA+5) * heightA + (rowA+0)] = vec1x8_0.s5;
    a_T[(colA+6) * heightA + (rowA+0)] = vec1x8_0.s6;
    a_T[(colA+7) * heightA + (rowA+0)] = vec1x8_0.s7;

		a_T[(colA+0) * heightA + (rowA+1)] = vec1x8_1.s0;
		a_T[(colA+1) * heightA + (rowA+1)] = vec1x8_1.s1;
   	a_T[(colA+2) * heightA + (rowA+1)] = vec1x8_1.s2;
    a_T[(colA+3) * heightA + (rowA+1)] = vec1x8_1.s3;
    a_T[(colA+4) * heightA + (rowA+1)] = vec1x8_1.s4;
    a_T[(colA+5) * heightA + (rowA+1)] = vec1x8_1.s5;
    a_T[(colA+6) * heightA + (rowA+1)] = vec1x8_1.s6;
    a_T[(colA+7) * heightA + (rowA+1)] = vec1x8_1.s7;

		a_T[(colA+0) * heightA + (rowA+2)] = vec1x8_2.s0;
		a_T[(colA+1) * heightA + (rowA+2)] = vec1x8_2.s1;
   	a_T[(colA+2) * heightA + (rowA+2)] = vec1x8_2.s2;
    a_T[(colA+3) * heightA + (rowA+2)] = vec1x8_2.s3;
    a_T[(colA+4) * heightA + (rowA+2)] = vec1x8_2.s4;
    a_T[(colA+5) * heightA + (rowA+2)] = vec1x8_2.s5;
    a_T[(colA+6) * heightA + (rowA+2)] = vec1x8_2.s6;
    a_T[(colA+7) * heightA + (rowA+2)] = vec1x8_2.s7;

		a_T[(colA+0) * heightA + (rowA+3)] = vec1x8_3.s0;
		a_T[(colA+1) * heightA + (rowA+3)] = vec1x8_3.s1;
   	a_T[(colA+2) * heightA + (rowA+3)] = vec1x8_3.s2;
    a_T[(colA+3) * heightA + (rowA+3)] = vec1x8_3.s3;
    a_T[(colA+4) * heightA + (rowA+3)] = vec1x8_3.s4;
    a_T[(colA+5) * heightA + (rowA+3)] = vec1x8_3.s5;
    a_T[(colA+6) * heightA + (rowA+3)] = vec1x8_3.s6;
    a_T[(colA+7) * heightA + (rowA+3)] = vec1x8_3.s7;

		a_T[(colA+0) * heightA + (rowA+4)] = vec1x8_4.s0;
		a_T[(colA+1) * heightA + (rowA+4)] = vec1x8_4.s1;
   	a_T[(colA+2) * heightA + (rowA+4)] = vec1x8_4.s2;
    a_T[(colA+3) * heightA + (rowA+4)] = vec1x8_4.s3;
    a_T[(colA+4) * heightA + (rowA+4)] = vec1x8_4.s4;
    a_T[(colA+5) * heightA + (rowA+4)] = vec1x8_4.s5;
    a_T[(colA+6) * heightA + (rowA+4)] = vec1x8_4.s6;
    a_T[(colA+7) * heightA + (rowA+4)] = vec1x8_4.s7;

		a_T[(colA+0) * heightA + (rowA+5)] = vec1x8_5.s0;
		a_T[(colA+1) * heightA + (rowA+5)] = vec1x8_5.s1;
   	a_T[(colA+2) * heightA + (rowA+5)] = vec1x8_5.s2;
    a_T[(colA+3) * heightA + (rowA+5)] = vec1x8_5.s3;
    a_T[(colA+4) * heightA + (rowA+5)] = vec1x8_5.s4;
    a_T[(colA+5) * heightA + (rowA+5)] = vec1x8_5.s5;
    a_T[(colA+6) * heightA + (rowA+5)] = vec1x8_5.s6;
    a_T[(colA+7) * heightA + (rowA+5)] = vec1x8_5.s7;

		a_T[(colA+0) * heightA + (rowA+6)] = vec1x8_6.s0;
		a_T[(colA+1) * heightA + (rowA+6)] = vec1x8_6.s1;
   	a_T[(colA+2) * heightA + (rowA+6)] = vec1x8_6.s2;
    a_T[(colA+3) * heightA + (rowA+6)] = vec1x8_6.s3;
    a_T[(colA+4) * heightA + (rowA+6)] = vec1x8_6.s4;
    a_T[(colA+5) * heightA + (rowA+6)] = vec1x8_6.s5;
    a_T[(colA+6) * heightA + (rowA+6)] = vec1x8_6.s6;
    a_T[(colA+7) * heightA + (rowA+6)] = vec1x8_6.s7;

		a_T[(colA+0) * heightA + (rowA+7)] = vec1x8_7.s0;
		a_T[(colA+1) * heightA + (rowA+7)] = vec1x8_7.s1;
   	a_T[(colA+2) * heightA + (rowA+7)] = vec1x8_7.s2;
    a_T[(colA+3) * heightA + (rowA+7)] = vec1x8_7.s3;
    a_T[(colA+4) * heightA + (rowA+7)] = vec1x8_7.s4;
    a_T[(colA+5) * heightA + (rowA+7)] = vec1x8_7.s5;
    a_T[(colA+6) * heightA + (rowA+7)] = vec1x8_7.s6;
    a_T[(colA+7) * heightA + (rowA+7)] = vec1x8_7.s7;
  }

}
