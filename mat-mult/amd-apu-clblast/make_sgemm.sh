#gcc -o benchmark_sgemm sgemm.c -I/opt/AMDAPPSDK-3.0/include/ -LOpenCL
gcc -o benchmark_sgemm sgemm.c -I/opt/AMDAPPSDK-3.0/include/  -lOpenCL -L../build/ -lclblast
