#pragma once

#define DEBUG 0

typedef struct Array_2d{
	float** array;
	int x;
	int y;
} array_2d;

void print(array_2d);
void tb_setup(array_2d*, array_2d*, array_2d*);
void convolve_2d(array_2d, array_2d, array_2d);

static int max(int a, int b){
	return a>=b?a:b;
}
