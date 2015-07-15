#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include <stdio.h>
#include <cudaconv2.cuh>
#include <errno.h>


#define IMG_SIZE 9216
#define FILTER_SIZE 1600

float * readMatrix_filter(char * filename, int nRows, int nCols);
float * readMatrix_img(char * filename, int nRows, int nCols);
float * genMatrix_img(int m, int n, float val);
void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor);

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
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread); // 64 / (8*4) = 4
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule); // 4 * 4 * ([0-71] %4) ==> 16 * [0-2] ==> 0. 32
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
    unsigned int last_idx_filter;
    unsigned int shift_idx_filter = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        shift_idx_filter += moduleIdx * numFilterColors * filterPixels * numFilters;
    }
    unsigned int last_idx_img;
    unsigned int shift2_idx_img;
    unsigned int shift_idx_img = blockColorIdx * imgPixels * imgStride + myImgIdx;
    //images += blockColorIdx * imgPixels * imgStride + myImgIdx;


    /*
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }
    */
    unsigned int last_idx_targets;
    unsigned int shift_idx_targets = moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;
    /*
    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;
    */

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
                            last_idx_filter = shift_idx_filter + (((oc+c) * filterPixels /*25*/ + p + p2) * numFilters);
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[last_idx_filter]; 
                            //shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels /*25*/ + p + p2) * numFilters /*64*/]; 

                            //if(filters[last_idx_filter] != 0.0)
                            //    shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = (float)last_idx_filter + (filters[last_idx_filter])*0.1;

                            //if(filters[last_idx_filter] == 0.0)
                            //    filters[last_idx_filter] = (float)moduleIdx + (oc+c)*100;

                            // (threadIdx.x)(threadIdx.y).(blockIdx.x)(blockIdx.y)
                            //shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0; 
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
                    shift2_idx_img = shift_idx_img + (imgStride * (oc * imgPixels + y * imgSizeX + x));
                    //float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                last_idx_img = shift2_idx_img + (c * imgStride * imgPixels + i * B_X);
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[last_idx_img];
                                //shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = (float)last_idx_img;

                                //if(images[last_idx_img] == 0.0)
                                //    images[last_idx_img] = (oc+c);
                                //else if(((int)images[last_idx_img])%100 == oc+c)
                                //    images[last_idx_img] = (oc+c);
                                //else
                                //    images[last_idx_img] = 9999.0;
                                //images[last_idx_img] = threadIdx.x*10 + threadIdx.y + blockIdx.y * 1000;

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
                        //prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y]; 
                        if(shFilters[i][threadIdx.y + f * B_Y] != 0.0)
                            prod[f][g] += 1.0; 
                        //if(shFilters[i][threadIdx.y + f * B_Y] != 0.0)
                        //    images[((int)(shImages[i][g * B_X + threadIdx.x]))] = shFilters[i][threadIdx.y + f * B_Y];
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
                    last_idx_targets = shift_idx_targets + (g * B_X + f * B_Y * numImages * numModules);
                    targets[last_idx_targets] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}

int main()
{
    int numImages = 128; //images.getNumCols();
    int numFilters = 64; //(64, 32) //filters.getNumCols();
    int imgSizeY = 6; //(12, 6)
    int imgSizeX = 6; //(12, 6) //imgPixels / imgSizeY;
    int filterSize = 3; //(5, 3) //int(sqrt(filterPixels));
    int paddingStart = (-1) * (filterSize/2); //(-2, -1)
    int moduleStride = 1; //one input acivation per one above(its center) neuron
    int numModulesY = 6; //(12, 6) usually same as imgSizeY
    int numModulesX = 6; //(12, 6)
    int imgStride = 128; //images.getStride(); usually same as numImages
    int numImgColors = 64; // (3, 32, 64)
    int numGroups = 1;
    float scaleTargets = 0.0;
    float scaleOutput = 1.0;
    bool conv = false; //(true, false)

    int nRowOfImg = (imgSizeX * imgSizeY) * numImgColors; //2304
    int nRowOfFilter = (filterSize * filterSize) * (numModulesX * numModulesY) * numImgColors; //20736

    float* img_data_host = readMatrix_img("data/local/zero_img.data", nRowOfImg, numImages); //  cur dir : /home/seungbin/npu/test/recompile-zero-out
    Matrix mat_img(img_data_host, nRowOfImg, numImages);
    NVMatrix images(mat_img, true);
    free(img_data_host);

    float* filter_data_host = readMatrix_filter("data/local/zero-out_filter.data", nRowOfFilter, numFilters); //"data/local/zero-out_zero_filter.data"
    //float* filter_data_host = readMatrix_filter("data/local/14th_neuron_0th_color_filter.data", nRowOfFilter, numFilters); //"data/local/zero-out_zero_filter.data"
    Matrix mat_filter(filter_data_host, nRowOfFilter, numFilters); 
    NVMatrix filters(mat_filter, true);//filters(FILTER_SIZE, FILTER_SIZE, false);
    free(filter_data_host);

    float* target_data_host = readMatrix_img("data/local/zero-out_targetInit.data", nRowOfImg, numImages); 
    Matrix mat_target(target_data_host, nRowOfImg, numImages); 
    NVMatrix targets(mat_target, true); 
    
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
        printf("gridDim(%d,%d,%d), blockDim(%d,%d,%d)\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
        //exit(0);
    }

    cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
    filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY,
        numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);

    targets.print(targets.getNumRows(), targets.getNumRows());
    //filters.print(filters.getNumRows(), filters.getNumRows());
    //images.print(images.getNumRows(), images.getNumRows());

    printf("\nfinish\n");

    cutilCheckMsg("filterActs: kernel execution failed");
}

float * readMatrix_filter(char * filename, int nRows, int nCols){

    float tmp;
    FILE *fp;
    float *full;
    full = (float *) malloc (nRows*nRows*sizeof(full[0]));

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file1\n");
        exit(1);
    }

    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nRows; ++j)
        {
            int ret = fscanf(fp, "%f ", &tmp);
            if(ret == 1){
                full[i*nRows + j] = tmp;
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

float * readMatrix_img(char * filename, int nRows, int nCols){

    float tmp;
    FILE *fp;
    float *full;
    full = (float *) malloc (nRows*nRows*sizeof(full[0]));

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file2\n");
        exit(1);
    }

    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nRows; ++j)
        {
            int ret = fscanf(fp, "%f ", &tmp);
            if(ret == 1){
                full[i*nRows + j] = tmp;
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