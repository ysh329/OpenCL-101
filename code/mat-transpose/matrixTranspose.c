#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>
#include <sys/time.h>

/* opencl */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define     MEM_SIZE            (128)
#define     MAX_SOURCE_SIZE     (0x1000000)

#define     MATRIX_WIDTH        (2)
#define     MATRIX_HEIGHT       (3)
#define     VECTOR_LEN          (3)


void init_mat(float *mat, int len, float setVal) {
    for (int idx = 0; idx < len; idx++)
		mat[idx] = setVal;
}

void rand_mat(float *mat, int len, int range) {
    if (range < 1) {
        printf("range value can't be less than 1.\n");
        exit(-1);
    }
    srand( (unsigned) time(0) );
    for (int idx = 0; idx < len; idx++)
        mat[idx] = rand() % range;
}

void print_mat(float *mat, int width, int height) {
	for (int r = 0; r < height; r++) {
	    for (int c = 0; c < width; c++) 
            printf("%.2f ", mat[c*height+r]);
        printf("\n");
    }
    printf("\n");
}

void print_vec(float *vec, int len) {
	for (int idx = 0; idx < len; idx++)
		printf("%.2f \n", vec[idx]);
}

void add_mat(float *a, float *b, float *res, int width, int height) {
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++)
            res[c*height + r] = a[c*height + r] + b[c*height + r];
}

void add_vec(float *a, float *b, float *res, int len) {
    for (int idx = 0; idx < len; idx++) 
        res[idx] = a[idx] + b[idx];
}

void transpose_mat(float *a, int width, int height, float *res) {
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++) {
            res[r*width + c] = a[c*height + r];
			//printf("res[%d, %d]:= a[%d, %d] = %.2f \n", r, c, c, r, a[c*height+r]);
			//printf("res[%d] := a[%d] = %.2f \n\n", r*width + c, c*height + r, a[c*height+r]);
		}
}

int equal_mat(float *a, float *b, int width, int height) {
	for (int c = 0; c < width; c++)
		for (int r = 0; r < height; r++) {
			if (a[c*height+r] - b[c*height+r] > 10e-8) {
				//printf("matrix a is NOT equal to matrix b\n");
				return 0;
			}
		}
	//printf("matrix a is equal to matrix b\n");
	return 1;
}

int equal_vec(float *a, float *b, int len) {
	for (int idx = 0; idx < len; idx++)
		if (a[idx] - b[idx] > 10e-8) {
			//printf("matrix a is NOT equal to matrix b\n");
			return 0;
		}
	//printf("matrix a is equal to matrix b\n");
	return 1;
}

void dotprod_mat(float *a, float *b, float *res, int len) {
	for (int idx; idx < len; idx++)
		res[idx] = a[idx] * b[idx];
}

int mult_mat(float *a, float *b, float *res, int M, int N, int K) {
	int i, j, p;
	init_mat(res, M*K, 0);
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++) 
			for (p = 0; p < K; p++) {
				res[j * M + i] += a[p * M + i] * b[j * K + p];
				printf("res[%d, %d] %.2f += a[%d, %d] %.2f * b[%d, %d] %.2f \n", i, j, res[j*M+i], i, p, a[p*M+i], p, j, b[j*K+p]);
			}
}


int main(void) {

    float *a, *b, *c;
    a = (float *) malloc (2 * 3 * sizeof(float));
    b = (float *) malloc (2 * 3 * sizeof(float));
    c = (float *) malloc (2 * 3 * sizeof(float));

    printf("a:\n");
    rand_mat(a, 2, 2);
    print_mat(a, 2, 3);

    printf("b:\n");
    rand_mat(b, 2, 3);
    print_mat(b, 2, 3);

    printf("c:\n");
    init_mat(c, 2*3, 0);
    print_mat(c, 2, 3);

    printf("[c := a + b] using add_mat \n");
    add_mat(a, b, c, 2, 3);
    print_mat(c, 2, 3);

    printf("[c := a + b] using add_vec\n");
    init_mat(c, 2*3, 0);
    add_vec(a, b, c, 2*3);
    print_mat(c, 2, 3);

    printf("[c^T]\n");
    float *c_t;
	c_t = (float *) malloc (2 * 3 * sizeof(float));
    transpose_mat(c, 2, 3, c_t);
	printf("finished transpose\n");
    print_mat(c_t, 3, 2);

	equal_mat(c, c, 2, 3);
	equal_mat(a, b, 2, 3);

	equal_vec(c, c, 2*3);
	equal_vec(a, b, 2*3);

	printf("[mult_mat]\n");
	//float *ma, *mb, *mc;
	//ma = (float *) malloc (2 * 2 * sizeof(float));
	//mb = (float *) malloc (2 * 2 * sizeof(float));
	//mc = (float *) malloc (2 * 2 * sizeof(float));

	float ma[4] = {2, 4, 1, 3};
	float mb[4] = {1, 1, 2, 0};
	float mc[4] = {0, 0, 0, 0};
	init_mat(mc, 2*2, 0);

	printf("ma:\n"); print_mat(ma, 2, 2);
	printf("mb:\n"); print_mat(mb, 2, 2);
	printf("mc:\n"); print_mat(mc, 2, 2);

	mult_mat(ma, mb, mc, 2, 2, 2);
	printf("[mult_mat] mc := ma * mb\n");
	print_mat(mc, 2, 2);

	printf("[dotprod_mat] mc := ma .* mb\n");
	dotprod_mat(ma, mb, mc, 2*2);
	print_mat(mc, 2, 2);


	/* GPU */
	cl_mem a_buff, b_buff, c_buff;
	a_buff = b_buff = c_buff = NULL;

	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platforms;

	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;

	cl_context context = NULL;
	cl_kernel kernel = NULL;
	cl_program program = NULL;	

	cl_command_queue command_queue = NULL;
	cl_int ret;

	/* Load the source code and containing the kernel */
	char string[MEM_SIZE];
	FILE *fp;
	char fileName[] = "./matrixTranspose.cl";
	char *source_str;
	size_t source_size;

	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "failed to load kernel.\n");
	}
	source_str = (char*) malloc (MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Platform
	ret = clGetPlatform(1, &platform_id, &ret_num_platform);	
	if (ret != CL_SUCCESS) {
		printf("failed to get platform ID.\n");
		goto error;
	}

	// Device
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (ret != CL_SUCCESS) {
		printf("failed to get device id.\n");
		goto error;
	}
	



error:
	;
	



    return 1;
}

