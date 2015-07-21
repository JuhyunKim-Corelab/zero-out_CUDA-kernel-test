#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>     /* assert */
#include <iostream>     // std::cout
#include <algorithm>    // std::stable_sort
#include <vector>       // std::vector
#include <string>
#include <string.h>
#define MAX_BUF 65536
#define BLOCKSIZE 32

//this code takes 'filter.reordered.data' file as an input,
//and generate load sequences for each neurons.

//g++ genLoadSeqForNeurons.cpp -o genLoadSeqForNeurons.exe
// usage: ./genLoadSeqForNeurons.exe [nMaxConnPerNeuron] [nNeuronPerFilter] [nFilter] [input file]
// e.g. : ./genLoadSeqForNeurons.exe 576 36 64

int main(int argc, char** argv){
    assert(argc == 5 && "4 arguments should be provided !!\n");
    unsigned nMaxConnPerNeuron = atoi(argv[1]);//576
    unsigned nNeuronPerFilter = atoi(argv[2]);//36
    unsigned nFilter = atoi(argv[3]);//64
    //printf("%d %d %d\n", nMaxConnPerNeuron, nNeuronPerFilter, nFilter);
    long nWeightVal = ((long)nMaxConnPerNeuron)*nNeuronPerFilter*nFilter;
    float *wm = (float *)malloc(nWeightVal*sizeof(wm[0])); //

    FILE *fp_WmatReordered;
    FILE *fp_loadSeq;
    char filename_WmatReordered[40] = "zero-out_filter.reordered.data";
    char filename_loadSeq[40] = "loadSeqForNeuron.data";
    if(argc == 5){
        char tmpstr[80] = "loadSeqForNeuron.data";
        char tmpstr2[80];
        int len= (int)strlen(argv[4]);
        strncpy(tmpstr2, argv[4], len-14);
        strcat(tmpstr2, tmpstr); strcpy(filename_loadSeq, tmpstr2);
        strcpy(filename_loadSeq, tmpstr2);
        strcpy(filename_WmatReordered, argv[4]);
    }

    if((fp_WmatReordered = fopen(filename_WmatReordered, "r+")) == NULL) {
        printf("file open Error(WmatReordered)\n");
        exit(1);
    }
    if((fp_loadSeq = fopen(filename_loadSeq, "w")) == NULL) {
        printf("file open Error(loadSeq)\n");
        exit(1);
    }

    int ret, ret_2;
    float tmp;
    long i;
    for (i = 0; i < nWeightVal; ++i){
        ret = fscanf(fp_WmatReordered, "%f ", &tmp);
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
    if((ret_2 = fscanf(fp_WmatReordered, "%f ", &tmp)) != EOF){
        printf("Error 2 EOF i:%ld, tmp:%f, ret:%d\n", i, tmp, ret_2);
        exit(0);
    }

    unsigned nNonzeroConn = 0;
    
    for (int filterIdx = 0; filterIdx < nFilter; ++filterIdx){
        for (int neuronIdx = 0; neuronIdx < nNeuronPerFilter; ++neuronIdx){
            for (int connIdx = 0; connIdx < nMaxConnPerNeuron; ++connIdx){
                if(wm[(neuronIdx*nMaxConnPerNeuron + connIdx)*nFilter + filterIdx] != 0){
                    fprintf(fp_loadSeq, "%d ", connIdx);
                    //printf("%d ", connIdx);
                    nNonzeroConn++;
                }
            }
            if (nNonzeroConn != 0)
                fprintf(fp_loadSeq, "\n");
            //printf("\n");
            nNonzeroConn=0;
        }
    }
    fprintf(fp_loadSeq, "\n");
    //printf("\n");


    printf("Done. \n");
    fclose(fp_loadSeq);
    fclose(fp_WmatReordered);
    return 0;
}


/*
    for (int i = 0; i < nz; ++i){
        printf("%f\t", nonzero[i]);
    }
    printf("\n");
    for (int i = 0; i < cnt; ++i){
        printf("%d\t", mapping[i]);
    }
    printf("\n");
*/