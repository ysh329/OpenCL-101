void matrixMultiplication(const int M,
        const int N,
        const int K, 
        const float a,
        const float b,
        float c) {
    float acc = 0.0;
    for (int rowC = 0; rowC < M; rowC++) {
        for (int colC = 0; colC < N; colC++) {
            acc = 0.0;
            for (int k; k < K; k++) {
                acc += a[k * M + rowC] * b[colC * K + k];
            }
            c[colC * M + rowC] = acc;
        }
    }
}
