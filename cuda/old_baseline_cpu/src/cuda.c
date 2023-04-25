//start with 2d convolve, also implements 3d convolve?
//3d convolve b/c images have depth (RGB) 
//author: ss
#include "cuda_n.h"
#include <stdlib.h>
#include <stdio.h>
#include "test.h"
#include <assert.h>
void convolve_2d(array_2d* pout, array_2d* pin, array_2d* pkernel){
	
	float** out = pout->array;
	int out_x = pout->x;
	int out_y = pout->y;
	
	float** in = pin->array;
	int in_x = pin->x;
	int in_y = pin->y;
	
	float** kernel = pkernel->array;
	int kernel_x = pkernel->x;
	int kernel_y = pkernel->y;
	
	assert(out_x == in_x); //if input padded
	assert(out_y == in_y); //if input padded
	
	
	pout->start_x = max(0, kernel_x-1);
	pout->start_y = max(0, kernel_y-1);
	float sum =0;
	for(int i = max(0, kernel_y-1); i < out_y; i++){
		for(int j = max(0, kernel_x-1); j < out_x; j++){
			for(int k = max(0, i-kernel_y); k <= i; k++){
				for(int l = max(0, j-kernel_x); l <= j; l++){
					if(k < in_y && i-k < kernel_y && 
					   l < in_x && j-l < kernel_x){
						if (DEBUG) printf("i=%d,j=%d,k=%d,l=%d\n", i,j,k,l);
						sum += in[k][l] * kernel[i-k][j-l];
					}
				}
			}
			out[i][j] = sum;
			sum=0;
		}
	}
	
	return;
}

void tb_setup(array_2d* out, array_2d* in, array_2d* kernel){
	
	out->array = (float**)malloc(sizeof(float*) * out->y);
	for(int i = 0; i < out->y; i++){
		out->array[i] = (float*)malloc(sizeof(float) * out->x);
		//init to 0 for now
		for(int j = 0; j < out->x; j++){
			out->array[i][j] = 0;
		}
	}
	
	in->array = (float**)malloc(sizeof(float*) * in->y);
	for(int i = 0; i < in->y; i++){
		in->array[i] = (float*)malloc(sizeof(float) * in->x);
		//init to 1 for now
		for(int j = 0; j < in->x; j++){
			in->array[i][j] = 1;
		}
	}
	
	kernel->array = (float**)malloc(sizeof(float*) * kernel->y);
	for(int i = 0; i < kernel->y; i++){
		kernel->array[i] = (float*)malloc(sizeof(float) * kernel->x);
		//init to s for now
		for(int j = 0; j < kernel->x; j++){
			kernel->array[i][j] = 1;
		}
	}
}

void print(array_2d a){
	for(int i = a.start_y; i <= a.end_y; i++){
		for(int k = a.start_x; k <= a.end_x; k++){
			printf("%f ",a.array[i][k]);
		}
		printf("\n");
	}
	printf("==================\n");
}
