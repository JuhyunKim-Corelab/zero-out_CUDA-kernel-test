#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::stable_sort
#include <vector>       // std::vector
#include <string>
#define MAX_BUF 65536
#define BLOCKSIZE 32

//generate sorting guide(Sort file and mapping) for reorder_neuron.cpp
//stable sort in ascending orders

//g++ genSortGuide.cpp -o genSortGuide.exe
class Neuron
{
public:
    unsigned int nNonzeroConn;
    unsigned int idx_original; //start from 0
    Neuron(unsigned int nnzc, unsigned int idx_o) : nNonzeroConn(nnzc), idx_original(idx_o) {};
};

bool compare_nzConn (Neuron i, Neuron j);

int main(){


    std::vector<Neuron> nvec;
    FILE *fp;
    FILE *fp_lfb;
    FILE *fp_target;
    FILE *fp_mapping;
    int ret, newline;
    char dummyString [MAX_BUF];
    unsigned int cnt = 0;
    unsigned int nzcnt = 0;
    float tmp;
    char filename_sort[40] = "nzConnSort.data";
    char filename_lfb[40] = "nLoadForBlocks.data";
    char filename_target[40] = "nzConnUnsort.data";//produced by target.data (count number of nz weight)
    char filename_mapping[40] = "nzConnMapping.data";

    if((fp_target = fopen(filename_target, "r+")) == NULL) {
        printf("file open Error1\n");
        exit(1);
    }
    if((fp_mapping = fopen(filename_mapping, "w")) == NULL) {
        printf("file open Error2\n");
        exit(1);
    }
    if((fp = fopen(filename_sort, "w")) == NULL) {
        printf("No such file\n");
        exit(1);
    }
    if((fp_lfb = fopen(filename_lfb, "w")) == NULL) {
        printf("No such file(nLoadForBlocks)\n");
        exit(1);
    }

    while(1){
        ret = fscanf(fp_target, "%f ", &tmp);
        if(ret == 1){
            nvec.push_back(Neuron((unsigned int)tmp, cnt));
            cnt++;
            if(fgets(dummyString, MAX_BUF, fp_target) == NULL){
                printf("BUF error !!!\n");
                exit(0);
            }
            //printf("dummy : %s\n", dummyString);
            /*
            if((newline = fgetc(fp_target)) != '\n'){
                printf("New line Error : %c\n", (char)newline);
                exit(0);
            }
            */
        }
        else{
            if(fgetc(fp_target) != EOF){
                printf("EOF Error\n");
                exit(0);
            }
            else
                break;
        }
    }

    printf("size: %ld\n", nvec.size()); 
    //for (int i = 0; i < nvec.size(); ++i){
    //    printf("%d, %d\n", nvec[i].nNonzeroConn, nvec[i].idx_original);
    //}

    unsigned threadCnt = 0; // will iteration across 0~(BLOCKSIZE-1) again and again
    std::stable_sort (nvec.begin(), nvec.end(), compare_nzConn);
    //printf("after sort, size: %ld\n", nvec.size()); 
    for (int i = 0; i < nvec.size(); ++i){
        if(nvec[i].nNonzeroConn != 0){ //if you wanna print 0 connection neuron, remove this if statement
            //fprintf(fp, "%d, %d\n", nvec[i].nNonzeroConn, nvec[i].idx_original);
            //printf("%d, %d\n", nvec[i].nNonzeroConn, nvec[i].idx_original);

            //print it to a separate files
            fprintf(fp, "%d\n", nvec[i].nNonzeroConn);
            if(threadCnt == 0)//number of load in first thread
                fprintf(fp_lfb, "%d\n", nvec[i].nNonzeroConn);
            fprintf(fp_mapping, "%d\n", nvec[i].idx_original);
            nzcnt++;
            threadCnt = (threadCnt+1)%BLOCKSIZE;
        } 
    }
    printf("number of living Neuron: %d, %d/32 = %d, %d mod 32 = %d \n", nzcnt, nzcnt, nzcnt/32, nzcnt,nzcnt%32);
    fclose(fp);
    fclose(fp_lfb);
    fclose(fp_target);
    fclose(fp_mapping);
    return 0;
}

bool compare_nzConn (Neuron i, Neuron j)
{
  return (i.nNonzeroConn > j.nNonzeroConn);
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