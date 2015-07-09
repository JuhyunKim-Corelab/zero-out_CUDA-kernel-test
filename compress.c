
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define MAX_ELEMENT_NUM 102401

//g++ compress.c -o compress.exe


int main(){

    float tmp;
    FILE *fp;
    FILE *fp_compressed;
    FILE *fp_mapping;
    float *nonzero;
    int *mapping;
    char filename[40] = "zero-out_filter.data";//"zero-out.data"
    char filename_compressed[40] = "zero-out_filter_COMPRESSED.data";//"zero-out.data"
    char filename_mapping[40] = "zero-out_filter_MAPPING.data";//"zero-out.data"
    long zero_cnt =0;
    long cnt =0;
    long nz = 0;
    nonzero = (float *) malloc (MAX_ELEMENT_NUM*sizeof(nonzero[0]));
    mapping = (int *) malloc (MAX_ELEMENT_NUM*sizeof(mapping[0]));
    

    if((fp_compressed = fopen(filename_compressed, "w")) == NULL) {
        printf("file open Error1\n");
        exit(1);
    }
    if((fp_mapping = fopen(filename_mapping, "w")) == NULL) {
        printf("file open Error2\n");
        exit(1);
    }

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file\n");
        exit(1);
    }

    for (int i = 0; i < MAX_ELEMENT_NUM; ++i)
    {
        int ret = fscanf(fp, "%f ", &tmp);
        if(ret == 1){
        	cnt++;
        	if (tmp == 0.0){
        		zero_cnt++;
                mapping[i] = -1;
        	}
        	else{
        	    nonzero[nz++] = tmp;
                mapping[i] = zero_cnt;
            }
        	
        }
        else if(errno != 0) {
            perror("scanf:");
            exit(0);
        } else if(ret == EOF) {
        	printf("reach EOF\n");
            break;
        } else {
            printf("No match.\n");
            exit(0);
        }
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

    
    for (int i = 0; i < nz; ++i){
    	fprintf(fp_compressed, "%.15f ", nonzero[i]);
    }
    fprintf(fp_compressed, "\n");

    for (int i = 0; i < cnt; ++i){
    	fprintf(fp_mapping,"%d ", mapping[i]);
    }
    fprintf(fp_mapping,"\n");
    

    printf("total: %ld, nonzero: %ld, zero: %ld\n", cnt, nz, zero_cnt);
    fclose(fp);
    fclose(fp_compressed);
    fclose(fp_mapping);
    return 0;
}
