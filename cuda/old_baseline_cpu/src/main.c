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
	
	out.start_x = 0;
	out.start_y = 0;
	out.end_x = out.x-1;
	out.end_y = out.y-1;
	in.start_x = 0;
	in.start_y = 0;
	in.end_x = in.x-1;
	in.end_y = in.y-1;
	kernel.start_x = 0;
	kernel.start_y = 0;
	kernel.end_x = kernel.x-1;
	kernel.end_y = kernel.y-1;

	tb_setup(&out, &in, &kernel);
	
	printf("======In=====\n");
	print(in);
	printf("======Kernel=====\n");
	print(kernel);
	
	convolve_2d(&out, &in, &kernel);
	
	printf("======Out=====\n");
	print(out);

	return 0;
}
