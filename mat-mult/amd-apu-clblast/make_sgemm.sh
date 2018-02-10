#gcc -o benchmark_sgemm sgemm.c -I/opt/AMDAPPSDK-3.0/include/ -LOpenCL

# AMD APU
#gcc -o benchmark_sgemm sgemm.c -I/opt/AMDAPPSDK-3.0/include/  -lOpenCL -L../build/ -lclblast

# RK3399 MALI
# /usr/local/lib/libclblast.so
# add export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH in ~/.bashrc
gcc -o benchmark_sgemm sgemm.c \
  -lOpenCL \
  -L/usr/local/lib  -lclblast
