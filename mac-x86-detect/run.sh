set -x
g++  -std=c++11 demo.cc -o demo.bin -framework OpenCL -DAPPLE
./demo.bin

