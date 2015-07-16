#include <stdio.h>
#include <assert.h>     /* assert */
#include <stdlib.h>
#include <errno.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::stable_sort
#include <vector>       // std::vector
#include <string>
#define MAX_BUF 65536

//
//take mapping file as an input, and reorder w matrix of each neurons
//g++ reorder_neuron.cpp -o reorder_neuron.exe
// usage: ./reorder_neuron.exe [nMaxConnPerNeuron] [nNeuronPerFilter] [nFilter]
// e.g. : ./reorder_neuron.exe 576 36 64
class NeuronWmatrix
{
public:
    unsigned nNonzeroConn;
    unsigned idx_original; //start from 0
    NeuronWmatrix(unsigned nnzc, unsigned idx_o) : nNonzeroConn(nnzc), idx_original(idx_o) {};
};

int main(int argc, char** argv){

    assert(argc == 4 && "3 arguments should be provided !!\n");
    unsigned nMaxConnPerNeuron = atoi(argv[1]);//576
    unsigned nNeuronPerFilter = atoi(argv[2]);//36
    unsigned nFilter = atoi(argv[3]);//64
    //printf("%d %d %d\n", nMaxConnPerNeuron, nNeuronPerFilter, nFilter);
    long nWeightVal = ((long)nMaxConnPerNeuron)*nNeuronPerFilter*nFilter;
    float *wm = (float *)malloc(nWeightVal*sizeof(wm[0])); //
    
    char filename_mapping[40] = "nzConnMapping.data";
    char filename_wmatrix[40] = "zero-out_filter.data";
    char filename_res[40] = "zero-out_filter.reordered.data";
    FILE *fp_mapping;
    FILE *fp_wmatrix;
    FILE *fp_res;
    if((fp_mapping = fopen(filename_mapping, "r+")) == NULL) {
        printf("No such file (mapping)\n");
        exit(1);
    }
    if((fp_wmatrix = fopen(filename_wmatrix, "r+")) == NULL) {
        printf("No such file (wmatrix)\n");
        exit(1);
    }

    int ret;
    float tmp;

    for (long i = 0; i < nWeightVal; ++i){
        ret = fscanf(fp_wmatrix, "%f ", &tmp);
        if(ret == 1){
            wm[i] = tmp;
        }
        else if(errno != 0) {
            perror("scanf:");
            exit(0);
        } else if(ret == EOF) {
            printf("Error EOF\n");
            exit(0);
        } else {
            printf("No match.\n");
            exit(0);
        }
    }
    if(fscanf(fp_wmatrix, "%f ", &tmp) != EOF){
        printf("Error 2 EOF\n");
        exit(0);
    }

    /*
    for (int i = 0; i < nNeuronPerFilter * nMaxConnPerNeuron; ++i){
        for (int j = 0; j < nFilter; ++j){
            printf("%f ", wm[j + nFilter*i]);
        }
        printf("\n");
    }
    */

    unsigned nNeuron = nFilter * nNeuronPerFilter;
    unsigned *mapping = (unsigned *)calloc(nNeuron, sizeof(mapping[0]));
    unsigned tmp2;
    unsigned idx = 0;
    while(1){
        ret = fscanf(fp_mapping, "%d ", &tmp2);
        if(ret == 1){
            mapping[idx] = tmp2;
            idx++;
        }
        else if(errno != 0) {
            perror("scanf:2");
            exit(0);
        } else if(ret == EOF) {
            printf("In mapping read, reach EOF\n");
            break;
        } else {
            printf("No match.\n");
            exit(0);
        }
    }

    fclose(fp_wmatrix);
    fclose(fp_mapping);

    printf("# of living neurons :%d\n", idx);
    //for (unsigned i = 0; i < nNeuron; ++i){
    //    printf("%d ", mapping[i]);
    //}
    printf("\n");


    float *wm_res = (float *)calloc(nWeightVal, sizeof(wm_res[0])); 
    //switch src, dst
    unsigned src = 13;
    unsigned dst = 0;
    unsigned src_p;
    unsigned dst_p;
    float t=0.0;
    for (unsigned k = 0; k < idx; ++k){
        dst = k;
        src = mapping[k];
        for (unsigned i = 0; i < nMaxConnPerNeuron; ++i){
            src_p = (src%nNeuronPerFilter)*(nFilter*nMaxConnPerNeuron) + (src/nNeuronPerFilter);
            dst_p = (dst%nNeuronPerFilter)*(nFilter*nMaxConnPerNeuron) + (dst/nNeuronPerFilter);
            wm_res[dst_p + i*nFilter] = wm[src_p + i*nFilter];
        }
    }

    //
    // write wm_res to file
    //
    if((fp_res = fopen(filename_res, "w")) == NULL) {
        printf("No such file (fp_res)\n");
        exit(1);
    }

    for (int i = 0; i < nNeuronPerFilter * nMaxConnPerNeuron; ++i){
        for (int j = 0; j < nFilter; ++j){
            //printf("%f ", wm_res[j + nFilter*i]);
            fprintf(fp_res, "%f ", wm_res[j + nFilter*i]);
        }
        //printf("\n");
        fprintf(fp_res, "\n");
    }
    fclose(fp_res);


}
