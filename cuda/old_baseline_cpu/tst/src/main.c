#include "cuda_n.h"
#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <assert.h>
//#include <cuda.h> dont even have this

int main(int argc, char* argv[]){
	
	assert(argc == 5);
	
	int inputs[4] = {};
	char *p;
	
	errno=0;
	for(int i = 1; i <= 4; i++){
		long conv = strtol(argv[i], &p, 10);
		
		if(errno != 0 || *p != '\0' || conv > INT_MAX || conv < INT_MIN)
			assert(0);
		
		inputs[i-1] = conv;
	}
	
	array_2d out;
	array_2d in;
	array_2d kernel;
	
	out.x = inputs[1];
	out.y = inputs[0];
	in.x = inputs[1];
	in.y = inputs[0];
	kernel.x = inputs[3];
	kernel.y = inputs[2];

	tb_setup(&out, &in, &kernel);
	
	printf("======In=====\n");
	print(in);
	printf("======Kernel=====\n");
	print(kernel);
	
	convolve_2d(out, in, kernel);
	
	printf("======Out=====\n");
	print(out);

	#ifdef DEBUG // defined in cuda_n.h
	// setup identity kernel
	// square kernel of 0s, except the center is 1
	#define IDENTITY_KERNEL_SIZE 3 // must be odd number to have a center value
	assert(IDENTITY_KERNEL_SIZE % 2 == 1);

	array_2d identity_kernel;
	identity_kernel.x = IDENTITY_KERNEL_SIZE;
	identity_kernel.y = IDENTITY_KERNEL_SIZE;
	int midpoint = IDENTITY_KERNEL_SIZE / 2 + 1;
	identity_kernel.array = (float**)malloc(sizeof(float*) * identity_kernel.y);

	for(int i = 0; i < identity_kernel.y; i++){
		identity_kernel.array[i] = (float*)malloc(sizeof(float) * identity_kernel.x);
		for(int j = 0; j < identity_kernel.x; j++){
			if (i == midpoint && j == midpoint){
				identity_kernel.array[i][j] = 1;
			}
			else {
				identity_kernel.array[i][j] = 0;
			}
		}
	}
	printf("======Identity Kernel=====\n");
	print(identity_kernel);
	tb_setup(&out, &in, &kernel);
	// convolving using the identity kernel should result in no changes
	convolve_2d(out, in, identity_kernel);

	for (int i = 0; i < out.y; i++){
		for(int j = 0; j < out.x; j++){
			assert(in.array[i][j] == out.array[i][j]);
		}
	}
	#endif

	return 0;
}
