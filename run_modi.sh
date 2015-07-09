#!bin/bash

/usr/local/cuda/bin/nvcc   -gencode=arch=compute_20,code=\"sm_20,compute_20\"   -m64 --compiler-options -fno-strict-aliasing --compiler-options '-fPIC'  -DNUMPY_INTERFACE -DMODELNAME=_ConvNet -DINITNAME=init_ConvNet -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I../../convnet/cuda-convnet-read-only/include -I../../convnet/cuda-convnet-read-only/include/common -I../../convnet/cuda-convnet-read-only/include/cudaconv2 -I../../convnet/cuda-convnet-read-only/include/nvmatrix -I../../convnet/cuda-convnet-read-only -I/usr/local/cuda/include -I/home/seungbin/NVIDIA_GPU_Computing_SDK/C/common/inc -I/home/seungbin/NVIDIA_GPU_Computing_SDK/shared//inc -DUNIX -O2 -c -o filter_Acts_XxY_sparse_modi.o filter_Acts_XxY_sparse_modi.cu

g++ -fPIC -Wl,-no-undefined  -m64 -o filter_Acts_XxY_sparse_modi.exe filter_Acts_XxY_sparse_modi.o ../../convnet/cuda-convnet-read-only/obj/x86_64/release/src/nvmatrix/nvmatrix.cu.o ../../convnet/cuda-convnet-read-only/obj/x86_64/release/src/common/matrix.cpp.o ../../convnet/cuda-convnet-read-only/obj/x86_64/release/src/nvmatrix/nvmatrix_kernels.cu.o  -lpthread -L/usr/lib/atlas-base -L/usr/local/cuda/lib64 -L/home/seungbin/NVIDIA_GPU_Computing_SDK/C/lib -L/home/seungbin/NVIDIA_GPU_Computing_SDK/shared/lib -lcblas -lpython2.7 -L/usr/local/cuda/lib64 -L/home/seungbin/NVIDIA_GPU_Computing_SDK/C/lib -L/home/seungbin/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -L/home/seungbin/NVIDIA_GPU_Computing_SDK/shared//lib -lcudart -lcublas -lcutil_x86_64 -lshrutil_x86_64

./filter_Acts_XxY_sparse_modi.exe > result.txt
#diff result.txt zero-out_target.data 

######