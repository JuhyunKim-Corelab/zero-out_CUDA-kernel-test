#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include <stdio.h>
#include <cudaconv2.cuh>
#include <errno.h>
//#include <cublas.h>

#define IMG_SIZE 9216
#define FILTER_SIZE 1600
#define FINISH(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (mapping_data_host)    free(mapping_data_host);      \
    if (d_mapping)   cudaFree(d_mapping);        \
    fflush (stdout);                                 \
    exit(1);                                        \
} while (0)

typedef int idx;

float * readMatrix_filter(char * filename, long *nElt);
idx * readMatrix_mapping(char * filename);
float * readMatrix_img(char * filename);
float * genMatrix_img(int m, int n, float val);
void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor);
void checkCublasError(const char* msg);

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache, //4,32,4,8,2
          bool scale, bool checkImgBounds> //false , false
__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
                                        idx* mappings,
                                       const int numImages, const int numFilters, //128, 64
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride, //1
                                       const int numModulesY, const int numModulesX, //12, 12
                                       const int imgStride, const int numImgColors, //128, 64
                                       const int numGroups,  //1
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) 
{
    __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups; //64 (64/1) //?
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups; //64
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup; //0

    const int numModules = numModulesX * numModulesY; //144
    const int blockColorIdx = numFilterColors * blockGroupIdx; // 64 * 0 = 0

    const int tidx = threadIdx.y * B_X + threadIdx.x; // linearized thread idx (0~127)

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride; // (-1 or -2) + ([0~143] / 12) * 1
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride; // (-1 or -2) + ([0~143] % 12) * 1

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    
    unsigned int shift_idx = blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        shift_idx = moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;

    /*
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }
    */

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    //    __shared__ int imgPos[]
    for (int oc = 0; oc < numFilterColors; oc += colorCache) 
    {
    // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) 
        {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread /*4*/) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            unsigned int last_idx = shift_idx + (((oc+c) * filterPixels /*25*/ + p + p2) * numFilters);
                            if(mappings[last_idx] == -1)
                                shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0.0;
                            else    
                                shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[last_idx - mappings[last_idx]]; 
                            //shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels /*25*/ + p + p2) * numFilters /*64*/]; 
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            /*
             * Load B_Y pixels from B_X*imgsPerThread images
             */
            const int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;
                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        //juhyun
                        //if(shFilters[i][threadIdx.y + f * B_Y])
                        //    prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];

                        //original
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y]; 
                    }
                }
            }
            __syncthreads();
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {//this will exec
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}

int main()
{
    float* img_data_host = readMatrix_img("zero-out_img.data");
    Matrix mat_img(img_data_host, 9216, 128);
    NVMatrix images(mat_img, true);
    free(img_data_host);

    long nElt = 0;
    float* filter_data_host = readMatrix_filter("zero-out_filter_COMPRESSED.data", &nElt); //float* filter_data_host = readMatrix_filter("zero-out_filter_COMPRESSED.data", &nElt);
    Matrix mat_filter(filter_data_host, 1600, 64); 
    NVMatrix filters(mat_filter, true);//filters(FILTER_SIZE, FILTER_SIZE, false);
    free(filter_data_host);

    printf("nElt = %ld\n", nElt);

    idx* mapping_data_host = readMatrix_mapping("zero-out_filter_MAPPING.data");
    cudaError_t cudaStat1;
    idx* d_mapping = 0;

    //cudaStat1 = cudaMalloc((void**)&d_mapping, 1600*64*sizeof(d_mapping[0]));
    //if (cudaStat1 != cudaSuccess) FINISH("cudaMalloc FAIL !!!");
    //cudaStat1 = cudaMemcpy(d_mapping, mapping_data_host, (size_t)(1600*64*sizeof(d_mapping[0])), cudaMemcpyHostToDevice);
    //if (cudaStat1 != cudaSuccess) FINISH("cudaMemcpy FAIL !!!");

    cublasAlloc(1600*64, sizeof(idx), (void**) &d_mapping);
    checkCublasError("!!!! device memory allocation error in MAPPING\n");
    cublasStatus status = cublasSetMatrix(64, 1600, sizeof(idx), mapping_data_host, 64, d_mapping, 64);
    if (status != CUBLAS_STATUS_SUCCESS) FINISH("cublasSetMatrix FAIL !!!");    

    float* target_data_host = readMatrix_img("zero-out_targetInit.data"); 
    Matrix mat_target(target_data_host, 9216, 128); 
    NVMatrix targets(mat_target, true);    

    int numImages = 128; //images.getNumCols();
    int numFilters = 64; //(64, 32) //filters.getNumCols();
    int imgSizeY = 12; //(12, 6)
    int imgSizeX = 12; //(12, 6) //imgPixels / imgSizeY;
    int filterSize = 5; //(5, 3) //int(sqrt(filterPixels));
    int paddingStart = -2; //(-2, -1)
    int moduleStride = 1;
    int numModulesY = 12; //(12, 6)
    int numModulesX = 12; //(12, 6)
    int imgStride = 128; //images.getStride();
    int numImgColors = 64;
    int numGroups = 1;
    float scaleTargets = 0.0;
    float scaleOutput = 1.0;
    bool conv = true; //(true, false)

    int imgsPerThread = 4;
    int numFilterColors = numImgColors / numGroups;      
    int numModules = numModulesY * numModulesX;
    int imgPixels = images.getNumRows()/numImgColors;
    int filterModuleMult = conv ? 1 : numModules;
    int numFiltersPerGroup = numFilters / numGroups;
    int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);

    if(1){
        assert(filterSize * filterSize == filterPixels);
        assert(filters.getNumRows() == filterModuleMult * numFilterColors * filterPixels);
        assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
        assert(numGroups == 1 || numFilterColors % 2 == 0);
        assert(numFilters % (16 * numGroups) == 0);
        assert(numImgColors % numGroups == 0);
        assert(images.getNumRows() == imgPixels * numImgColors);
        assert(imgSizeY * imgSizeX == imgPixels);
        // These routines don't handle the case when only part of the image is visited in the convolution
        assert(paddingStart <= 0);
        assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
        assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
        assert(moduleStride <= filterSize);
        assert(!images.isTrans());
        assert(!filters.isTrans());
        assert(!targets.isTrans());
        assert(filters.isContiguous());
        assert(targets.isContiguous());
    }
    
    dim3 blocks = numFiltersPerGroup % 32 == 0 ? dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8))
                                               : dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));
    dim3 threads(32, 4);
    bool checkImgBounds = numImages % (32*imgsPerThread) != 0;
    
    if (scaleTargets == 0) {
        ;//targets.resize(numFilters * numModules, numImages);
    } else {
        assert(targets.getNumRows() == numFilters * numModules);
        assert(targets.getNumCols() == numImages);
    }


    if(1)
    {    ;
        //printf("#################\n");
        //printf("image >> rows: %d, cols: %d, stride: %d, isTrans?:%d, ownsData?:%d\n"
        //    , images.getNumRows(), images.getNumRows(), images.getStride(), images.isTrans(), !images.isView());
        //printf("filters>> rows: %d, cols: %d, stride: %d, isTrans?:%d, ownsData?:%d\n"
        //    , filters.getNumRows(), filters.getNumRows(), filters.getStride(), filters.isTrans(), !filters.isView());

        //images.print(images.getNumRows(), images.getNumRows());
        //filters.print(filters.getNumRows(), filters.getNumRows());
        //targets.print(targets.getNumRows(), targets.getNumRows());
        //printf("<<<<<<<<<<<<<<<<<<<<<<<<<\n");
        //printf("gridDim(%d,%d,%d), blockDim(%d,%d,%d)\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
        //exit(0);
    }

    cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
    filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
        d_mapping,
        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY,
        numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);

    targets.print(targets.getNumRows(), targets.getNumRows());

    free(mapping_data_host);

    printf("\nfinish\n");

    cutilCheckMsg("filterActs: kernel execution failed");
}

float * readMatrix_filter(char * filename, long *nElt){

    float tmp;
    FILE *fp;
    float *full;
    full = (float *) malloc (1600*64*sizeof(full[0]));
    long cnt = 0;
    int ret;

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file\n");
        exit(1);
    } 

    for (int i = 0; i < 1600; ++i)
    {
        for (int j = 0; j < 64; ++j)
        {
            ret = fscanf(fp, "%f ", &tmp);
            if(ret == 1){
                full[i*64 + j] = tmp;
                //printf("%.15f\n", tmp);
                cnt++;
            }
            else if(errno != 0) {
                    perror("scanf:");
                    break;
            } else if(ret == EOF) {
                printf("reach EOF in readMatrix_filter.\n");
                break;
            } else {
                printf("No match.\n");
                exit(0);
            }
        }
        if(ret == EOF) break;
    }
    *nElt = cnt;

    return full;//full_dev
}


idx * readMatrix_mapping(char * filename){
    idx tmp;
    FILE *fp;
    idx *full;
    full = (idx *) malloc (1600*64*sizeof(full[0]));
    int ret;

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file\n");
        exit(1);
    } 

    for (int i = 0; i < 1600; ++i)
    {
        for (int j = 0; j < 64; ++j)
        {
            ret = fscanf(fp, "%u ", &tmp);
            if(ret == 1){
                full[i*64 + j] = tmp;
            }
            else if(errno != 0) {
                    perror("scanf:");
                    break;
            } else if(ret == EOF) {
                printf("reach EOF in readMatrix_mapping.\n");
                break;
            } else {
                printf("No match.\n");
                exit(0);
            }
        }
        if(ret == EOF) break;
    }

    return full;//full_dev
}


float * readMatrix_img(char * filename){

    float tmp;
    FILE *fp;
    float *full;
    full = (float *) malloc (9216*128*sizeof(full[0]));
    
    /*
    float *test;
    test = (float *) malloc (2*3*sizeof(test[0]));
    test[0] = 1; test[1] = 2; test[2] = 3;
    test[3] = 4; test[4] = 5; test[5] = 6;
    Matrix mat(test, 2, 3);
    mat.print();
    */

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file\n");
        exit(1);
    }

    for (int i = 0; i < 9216; ++i)
    {
        for (int j = 0; j < 128; ++j)
        {
            int ret = fscanf(fp, "%f ", &tmp);
            if(ret == 1){
                full[i*128 + j] = tmp;
                //printf("%.15f\n", tmp);
            }
            else if(errno != 0) {
                    perror("scanf:");
                    break;
            } else if(ret == EOF) {
                //printf("finish.\n");
                break;
            } else {
                printf("No match.\n");
                exit(0);
            }
        }
    }

    return full;//full_dev
}

float * genMatrix_img(int m, int n, float val){
    float *full;
    full = (float *)malloc(m*n*sizeof(full[0]));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            full[j+i*n] = val;
        }
    }
    return full;
}

void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor){
    
    //printf("$$$$$$$$$ RESULT $$$$$$$$$$$$$$$\n");
    for (int y = 0; y < mR; ++y)
    {
        for (int x = 0; x < nR; ++x)
        {
            if(x<real_nR && y<real_mR){
                //if(result[nR*y + x] != -1.0)
                    printf("%.4f\t", result[nR*y + x]);
            }
        }
        printf("\n");
    }
    //printf("============END==========\n");
}


void checkCublasError(const char* msg) {
    cublasStatus status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, msg, NULL);
        exit(EXIT_FAILURE);
    }
}