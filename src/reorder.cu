#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include <stdio.h>
#include <cudaconv2.cuh>
#include <errno.h>

#define MAX_BUFSIZE 65536
#define IMG_SIZE 9216
#define FILTER_SIZE 1600
#define N_LIVING_NEURON 2027
#define GRID_DIM_X 64
#define BLOCK_SIZE 32

// // ceil(N_LIVING_NEURON/BLOCK_SIZE) = (N_LIVING_NEURON - 1)/BLOCK_SIZE +1 = (2027 - 1)/32 + 1

float * readMatrix_filter(char * filename, int nRows, int nCols);
float * readMatrix_img(char * filename, int nRows, int nCols);
float * genMatrix_img(int m, int n, float val);
unsigned * readMapping(char * filename, unsigned nLivingNeuron);
unsigned * readnLoadForBlock(char * filename, unsigned nBlocks);
int ** readLoadSeqs(char * filename, unsigned nBlocks, unsigned* nLoadForBlocks);
void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor);
int numberOfDigits(int n);


__global__ void reorderedFilters(float* images, float* filters, float* targets,
									   unsigned* mapping, unsigned* nLoadForBlocks, int** loadSeqs,
                                       const int numImages, const int numFilters, //128, 64
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride, //1
                                       const int numModulesY, const int numModulesX, //12, 12
                                       const int imgStride, const int numImgColors, //128, 64
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

        float* img_data_host = readMatrix_img("../data/local/zero-out_img.data", nRowOfImg, numImages); //  cur dir : /home/seungbin/npu/test/recompile-zero-out
        Matrix mat_img(img_data_host, nRowOfImg, numImages);
        NVMatrix images(mat_img, true);
        free(img_data_host);

        float* filter_data_host = readMatrix_filter("../data/local/zero-out_filter.reordered.data", nRowOfFilter, numFilters); //"../data/local/zero-out_zero_filter.data"
        //float* filter_data_host = readMatrix_filter("../data/local/14th_neuron_0th_color_filter.data", nRowOfFilter, numFilters); //"../data/local/zero-out_zero_filter.data"
        Matrix mat_filter(filter_data_host, nRowOfFilter, numFilters); 
        NVMatrix filters(mat_filter, true);//filters(FILTER_SIZE, FILTER_SIZE, false);
        free(filter_data_host);

        float* target_data_host = readMatrix_img("../data/local/zero_img.data", nRowOfImg, numImages); //zero-out_targetInit.data
        Matrix mat_target(target_data_host, nRowOfImg, numImages); 
        NVMatrix targets(mat_target, true); 

        unsigned* mapping_h = readMapping("../data/local/nzConnMapping.data", N_LIVING_NEURON); 
        unsigned* mapping_d = 0;
        cudaError_t cudaStat1;
        cudaStat1 = cudaMalloc((void**)&mapping_d, N_LIVING_NEURON*sizeof(mapping_d[0]));
        if (cudaStat1 != cudaSuccess) {
            printf("Device malloc failed (mapping_d)");
            exit(1);
        }
        cudaStat1 = cudaMemcpy(mapping_d, mapping_h, (size_t)(N_LIVING_NEURON*sizeof(mapping_d[0])), cudaMemcpyHostToDevice);
        if (cudaStat1 != cudaSuccess) {
            printf("cudaMemcpy failed (mapping_d) %d  cudaErrorInvalidValue :%d", cudaStat1, cudaErrorInvalidValue );
            exit(1);
        }
        free(mapping_h);

        unsigned* nLoadforBlocks_h = readnLoadForBlock("../data/local/nLoadForBlocks.data", GRID_DIM_X);
        unsigned* nLoadforBlocks_d = 0;
        cudaStat1 = cudaMalloc((void**)&nLoadforBlocks_d, GRID_DIM_X*sizeof(nLoadforBlocks_d[0]));
        if (cudaStat1 != cudaSuccess) {
            printf("Device malloc failed (nLoadforBlocks_d)");
            exit(1);
        }
        cudaStat1 = cudaMemcpy(nLoadforBlocks_d, nLoadforBlocks_h, (size_t)(GRID_DIM_X*sizeof(nLoadforBlocks_d[0])), cudaMemcpyHostToDevice);
        if (cudaStat1 != cudaSuccess) {
            printf("cudaMemcpy failed (nLoadforBlocks_d) %d  cudaErrorInvalidValue :%d", cudaStat1, cudaErrorInvalidValue );
            exit(1);
        }
        
        int **loadSeqs_h = readLoadSeqs("../data/local/loadSeqForNeuron.data", GRID_DIM_X, nLoadforBlocks_h);
        int **loadSeqs_d = 0;
        cudaStat1 = cudaMalloc((void**)&loadSeqs_d, N_LIVING_NEURON*sizeof(loadSeqs_d[0]));
        if (cudaStat1 != cudaSuccess) {
            printf("Device malloc failed (loadSeqs_d)");
            exit(1);
        }
        int** loadSeqs_support = (int **)malloc(sizeof(int*)*N_LIVING_NEURON);
        cudaStat1 = cudaMemcpy(loadSeqs_support, loadSeqs_d, N_LIVING_NEURON*sizeof(int*), cudaMemcpyDeviceToHost);
        if (cudaStat1 != cudaSuccess) {
            printf("cudaMemcpy failed (loadSeqs_support) %d  cudaErrorInvalidValue :%d", cudaStat1, cudaErrorInvalidValue );
            exit(1);
        }
        for (unsigned neuronIdx = 0; neuronIdx < N_LIVING_NEURON; ++neuronIdx){
        	cudaStat1 = cudaMalloc((void**) &loadSeqs_support[neuronIdx], (nLoadforBlocks_h[ neuronIdx/BLOCK_SIZE ])*sizeof(int));
	        if (cudaStat1 != cudaSuccess) {
	            printf("Device malloc failed (loadSeqs_support[%d])", neuronIdx);
	            exit(1);
	        }
	        cudaStat1 = cudaMemcpy(loadSeqs_support[neuronIdx], loadSeqs_h[neuronIdx],
        					(size_t)((nLoadforBlocks_h[ neuronIdx/BLOCK_SIZE ])*sizeof(int)), cudaMemcpyHostToDevice);
	        if (cudaStat1 != cudaSuccess) {
	            printf("cudaMemcpy failed (loadSeqs_d) %d %d", cudaStat1, neuronIdx );
	            exit(1);
	        }
	        cudaStat1 = cudaMemcpy(loadSeqs_d+neuronIdx, &(loadSeqs_support[neuronIdx]), 1*sizeof(int*), cudaMemcpyHostToDevice);
	        if (cudaStat1 != cudaSuccess) {
	            printf("cudaMemcpy failed (!!!!!) %d  cudaErrorInvalidValue :%d", cudaStat1, cudaErrorInvalidValue );
	            exit(1);
	        }
        }
        free(nLoadforBlocks_h);
        free(loadSeqs_h);
        
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
    dim3 blocks (GRID_DIM_X, 1, 1); // ceil(2027/32) = (2027 - 1)/32 + 1
    dim3 threads(32, 1, 1);
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

    //for (int i = 0; i < GRID_DIM_X; ++i){
        /* code */
    //}
    
    //cudaFuncCachePreferNone//cudaFuncCachePreferShared//cudaFuncCachePreferL1
    cudaFuncSetCacheConfig(reorderedFilters, cudaFuncCachePreferL1);
    reorderedFilters <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(),
    	mapping_d, nLoadforBlocks_d, loadSeqs_d,
        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY,
        numModulesX, imgStride, numImgColors, scaleTargets, scaleOutput, conv);

    //targets.print(targets.getNumRows(), targets.getNumRows());
    //filters.print(filters.getNumRows(), filters.getNumRows());
    targets.print(targets.getNumRows(), targets.getNumRows());

    printf("\nfinish\n");

    cudaFree(mapping_d);
    cutilCheckMsg("filterActs: kernel execution failed");
}

__global__ void reorderedFilters(float* images, float* filters, float* targets,
									   unsigned* mapping, unsigned* nLoadForBlocks, int** loadSeqs,
                                       const int numImages, const int numFilters, //128, 64
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride, //1
                                       const int numModulesY, const int numModulesX, //6, 6
                                       const int imgStride, const int numImgColors, //128, 64
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv)
{
    //__shared__ float shFilters[Y][X];
    //__shared__ float shImages[Y][X];
    const int nMaxConnPerNeuron = (filterSize*filterSize) * numImgColors;
    const int neuronIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(neuronIdx >= N_LIVING_NEURON) return; //for dead neuron of last block
    const int nNeuronPerFilter = numModulesX * numModulesY;//36
    const int neuronIdx_old = mapping[neuronIdx];
    int* loadSeqs_thisNeuron = loadSeqs[neuronIdx];//loadSeqs[neuronIdx];

    float privWeight[576];//privWeight[nMaxConnPerNeuron]; literal because "nMaxConnPerNeuron" should be known in compile time
    float privAct[576];//equal to Weights
    float prod[128];
    for (int i = 0; i < numImages; ++i){
        prod[i] =0.0;
    }

    __shared__ unsigned nLoads;
    nLoads = nLoadForBlocks[ blockIdx.x ]; //what happen if all thread in a block load same memory location?

    /*
     * (weight load) initialization Phase
     */
    const unsigned loc = (neuronIdx%nNeuronPerFilter)*(numFilters*nMaxConnPerNeuron) + (neuronIdx/nNeuronPerFilter);//for first weight in that neuron
    for (int i = 0; i < nMaxConnPerNeuron; ++i){
        //(neuronIdx%nNeuronPerFilter)*(numFilters*nMaxConnPerNeuron) + (neuronIdx/nNeuronPerFilter);
        privWeight[i] = filters[loc + numFilters*i];
    }

    /*
     * (activation) Load Phase
     */
    int act_idx;
    const int center = neuronIdx_old%nNeuronPerFilter;//center : neuronIdx_old w/o color info
    //check padding condition
    //   * 1 *
    //   2   3
    //   * 4 *
    int padding1 = 0;
    int padding2 = 0;
    int padding3 = 0;
    int padding4 = 0;
    for (int i = 0; i < filterSize/2; ++i){
        padding1 += (int)(center/imgSizeX == i);
        padding2 += (int)(center%imgSizeX == i);
        padding3 += (int)((imgSizeX - 1) - center%imgSizeX == i);
        padding4 += (int)((imgSizeX - 1) - center/imgSizeX == i);
    }
    const int upperLeft = center - ((filterSize)/2) - imgSizeX*((filterSize)/2);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~iterate for 128 img~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for(int img_idx = 0; img_idx <numImages; img_idx++){
    	for (int i = 0; i < nLoads; ++i){ //loadSeqs_thisNeuron[i]: this will iterate actually.
    		int actLoadIdx = loadSeqs_thisNeuron[i];
    		if(actLoadIdx == -1) break;
    		int c = actLoadIdx/(filterSize*filterSize); //color
    		int y = (actLoadIdx%(filterSize*filterSize))/filterSize; // y idx in 3x3 filter
    		int x = (actLoadIdx%(filterSize*filterSize))%filterSize; // x idx in 3x3 filter
    		if(y >= padding1 && (filterSize - 1) - y >= padding4 ){
    			if(x >= padding2 && (filterSize - 1) - x >= padding3 ){
	    			privAct[c*(filterSize*filterSize) + (y*filterSize + x)] =
	    				images[(c*(nNeuronPerFilter) + upperLeft + y*imgSizeX + x)*numImages + img_idx];
	    				// [()*numImages + "n-th image"] // n:[0-127]
    			}
    			else{
    				privAct[c*(filterSize*filterSize) + (y*filterSize + x)] = 0.0;	
    			}
    		}
    		else{
    			privAct[c*(filterSize*filterSize) + (y*filterSize + x)] = 0.0;
    		}
    	}

        /*
        * Computation Phase
        */
        for (int i = 0; i <nMaxConnPerNeuron; ++i){
         prod[img_idx] += privAct[i] * privWeight[i];
        }  
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /*
    * Store Phase
    */
    for (int i = 0; i < numImages; ++i){
        targets[(neuronIdx_old)*numImages + i] = prod[i];//((float)((long)loadSeqs));//(float)(*loadSeqs_thisNeuron);
        //targets[(neuronIdx_old)*numImages + i] = prod[i]; //target[()*numImages + "n-th image"]
    }
    
}



int ** readLoadSeqs(char * filename, unsigned nBlocks, unsigned* nLoadForBlocks){
    int tmp;
    FILE *fp;
    int **res = NULL;
    int ret, newline;
    //res = (int *)malloc(nBlocks*sizeof(res[0]));
    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file (readLoadSeqs)\n");
        exit(1);
    }
    res = (int **)malloc(N_LIVING_NEURON*sizeof(res[0]));

    for (unsigned neuronIdx = 0; neuronIdx < N_LIVING_NEURON; ++neuronIdx){
        res[neuronIdx] = (int *)malloc(nLoadForBlocks[ neuronIdx/BLOCK_SIZE ]*sizeof(res[neuronIdx][0]));
        char line[MAX_BUFSIZE];
        char *pos = line;
        if(fgets (line, MAX_BUFSIZE, fp) == NULL){
            printf("fgets error!! \n");
            exit(0);
        }

        int i;
        for (i = 0; i < nLoadForBlocks[ neuronIdx/BLOCK_SIZE ]; ++i){
            ret = sscanf(pos, "%d ", &tmp);
            if(ret == 1){
                //printf("%d ", tmp);
                res[neuronIdx][i] = tmp;
                pos = pos + (1 + numberOfDigits(tmp));
            }
            else if(errno != 0) {
                    perror("scanf:");
                    break;
            } else if(ret == EOF) {
                //printf(" line finish. %d %d\n", i, nLoadForBlocks[ neuronIdx/BLOCK_SIZE ] - i);
                break;
            } else {
                printf("No match.\n");
                exit(0);
            }
        }
        for (; i < nLoadForBlocks[ neuronIdx/BLOCK_SIZE ]; ++i){
        	//printf("%d ", -1);
            res[neuronIdx][i] = -1; //for dummy load
        }
        //printf("\n");
    }

    fclose(fp);
    return res;
}



unsigned * readnLoadForBlock(char * filename, unsigned nBlocks){
    unsigned tmp;
    FILE *fp;
    unsigned *res;
    int ret;
    res = (unsigned *)malloc(nBlocks*sizeof(res[0]));
    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file (readnLoadForBlock)\n");
        exit(1);
    }

    for (unsigned i = 0; i < nBlocks; ++i){
        ret = fscanf(fp, "%d ", &tmp);
        if(ret == 1){
                res[i] = tmp;
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
    fclose(fp);
    return res;
}

unsigned * readMapping(char * filename, unsigned nLivingNeuron){
    unsigned tmp;
    FILE *fp;
    unsigned *res;
    int ret;
    res = (unsigned *)malloc(nLivingNeuron*sizeof(res[0]));
    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file (readMapping)\n");
        exit(1);
    }

    for (unsigned i = 0; i < nLivingNeuron; ++i){
        ret = fscanf(fp, "%d ", &tmp);
        if(ret == 1){
                res[i] = tmp;
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
    fclose(fp);
    return res;
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

int numberOfDigits(int n){
	if(n/100000 != 0)
		return 6;
	else if(n/10000 != 0)
		return 5;
	else if(n/1000 != 0)
		return 4;
	else if(n/100 != 0)
		return 3;
	else if(n/10 != 0)
		return 2;
	else
		return 1;
}












//grave yard
/*

			for (int c = 0; c < numImgColors; ++c){
            act_idx = 0;
            for (int y = 0; y < filterSize; ++y){
                if(y >= padding1 && (filterSize - 1) - y >= padding4 ){
                    for (int x = 0; x < filterSize; ++x){
                        if(x >= padding2 && (filterSize - 1) - x >= padding3 ){
                            privAct[c*(filterSize*filterSize) + act_idx] = images[(c*(nNeuronPerFilter) + upperLeft + y*imgSizeX + x)*numImages + img_idx];// [()*numImages + "n-th image"] // n:[0-127]
                            act_idx++;
                            //c = actLoadIdx/(filterSize*filterSize), y = (actLoadIdx%(filterSize*filterSize))/filterSize, x = (actLoadIdx%(filterSize*filterSize))%filterSize
                        }
                        else{
                            privAct[c*(filterSize*filterSize) + act_idx] = 0.0;
                            act_idx++;
                        }     
                    }
                }
                else{
                    for (int x = 0; x < filterSize; ++x){
                        privAct[c*(filterSize*filterSize) + act_idx] = 0.0;
                        act_idx++;
                    }
                }
            }
        }


*/