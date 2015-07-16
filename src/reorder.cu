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

__global__ void reorderedFilters(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters, //128, 64
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride, //1
                                       const int numModulesY, const int numModulesX, //12, 12
                                       const int imgStride, const int numImgColors, //128, 64
                                       const int numGroups,  //1
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv);

int main()
{   
    // ============ Declarations ============ //
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

        float* img_data_host = readMatrix_img("../data/local/zero_img.data", nRowOfImg, numImages); //  cur dir : /home/seungbin/npu/test/recompile-zero-out
        Matrix mat_img(img_data_host, nRowOfImg, numImages);
        NVMatrix images(mat_img, true);
        free(img_data_host);

        float* filter_data_host = readMatrix_filter("../data/local/zero-out_filter.data", nRowOfFilter, numFilters); //"../data/local/zero-out_zero_filter.data"
        //float* filter_data_host = readMatrix_filter("../data/local/14th_neuron_0th_color_filter.data", nRowOfFilter, numFilters); //"../data/local/zero-out_zero_filter.data"
        Matrix mat_filter(filter_data_host, nRowOfFilter, numFilters); 
        NVMatrix filters(mat_filter, true);//filters(FILTER_SIZE, FILTER_SIZE, false);
        free(filter_data_host);

        float* target_data_host = readMatrix_img("../data/local/zero-out_targetInit.data", nRowOfImg, numImages); 
        Matrix mat_target(target_data_host, nRowOfImg, numImages); 
        NVMatrix targets(mat_target, true); 
        
        int imgsPerThread = 4;
        int numFilterColors = numImgColors / numGroups;      
        int numModules = numModulesY * numModulesX;
        int imgPixels = images.getNumRows()/numImgColors;
        int filterModuleMult = conv ? 1 : numModules;
        int numFiltersPerGroup = numFilters / numGroups;
        int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);
    // ============ Assertion ============ //
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
    
    //cudaFuncCachePreferNone//cudaFuncCachePreferShared//cudaFuncCachePreferL1
    //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
    //filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
    //    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY,
    //    numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);

    //targets.print(targets.getNumRows(), targets.getNumRows());
    //filters.print(filters.getNumRows(), filters.getNumRows());
    //images.print(images.getNumRows(), images.getNumRows());

    printf("\nfinish\n");

    cutilCheckMsg("filterActs: kernel execution failed");
}

__global__ void reorderedFilters(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters, //128, 64
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride, //1
                                       const int numModulesY, const int numModulesX, //12, 12
                                       const int imgStride, const int numImgColors, //128, 64
                                       const int numGroups,  //1
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv)
{
    
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