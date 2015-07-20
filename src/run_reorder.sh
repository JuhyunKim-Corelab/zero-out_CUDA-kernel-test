#!bin/bash
CUDA_CONVNET_DIR=/home/seungbin/npu/convnet

/usr/local/cuda/bin/nvcc   -gencode=arch=compute_20,code=\"sm_20,compute_20\"   -m64 --compiler-options -fno-strict-aliasing --compiler-options '-fPIC'  -DNUMPY_INTERFACE -DMODELNAME=_ConvNet -DINITNAME=init_ConvNet -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I$CUDA_CONVNET_DIR/reorder_cuda-convnet/include -I$CUDA_CONVNET_DIR/reorder_cuda-convnet/include/common -I$CUDA_CONVNET_DIR/reorder_cuda-convnet/include/cudaconv2 -I$CUDA_CONVNET_DIR/reorder_cuda-convnet/include/nvmatrix -I$CUDA_CONVNET_DIR/reorder_cuda-convnet -I/usr/local/cuda/include -I/home/seungbin/NVIDIA_GPU_Computing_SDK/C/common/inc -I/home/seungbin/NVIDIA_GPU_Computing_SDK/shared//inc -DUNIX -O2 -c -o ../obj/reorder.o reorder.cu

g++ -fPIC -Wl,-no-undefined  -m64 -o ../obj/reorder.exe ../obj/reorder.o $CUDA_CONVNET_DIR/reorder_cuda-convnet/obj/x86_64/release/src/nvmatrix/nvmatrix.cu.o $CUDA_CONVNET_DIR/reorder_cuda-convnet/obj/x86_64/release/src/common/matrix.cpp.o $CUDA_CONVNET_DIR/reorder_cuda-convnet/obj/x86_64/release/src/nvmatrix/nvmatrix_kernels.cu.o  -lpthread -L/usr/lib/atlas-base -L/usr/local/cuda/lib64 -L/home/seungbin/NVIDIA_GPU_Computing_SDK/C/lib -L/home/seungbin/NVIDIA_GPU_Computing_SDK/shared/lib -lcblas -lpython2.7 -L/usr/local/cuda/lib64 -L/home/seungbin/NVIDIA_GPU_Computing_SDK/C/lib -L/home/seungbin/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -L/home/seungbin/NVIDIA_GPU_Computing_SDK/shared//lib -lcudart -lcublas -lcutil_x86_64 -lshrutil_x86_64

./../obj/reorder.exe > ../data/tmp_result.data
#diff result.txt target.data 
#rm result.txt
