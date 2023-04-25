#include "convolution.hpp"
#include <iostream>

#define o(x) std::cout << #x << x << std::endl
int main(){
	
	std::vector<unsigned long> dimensions = {3, 1, 2};
	std::vector<int> values = {1, 2, 3, 4, 5, 6};
	
	n_at::Dimension d(dimensions);
	n_at::Tensor<int> t(values, d);
	
	std::cout << t << std::endl;
	
	n_at::Tensor<int> slice = t.getSlice2d(1);
	std::cout << slice << std::endl;
	slice = t.getSlice2d(2);
	std::cout << slice << std::endl;
	
	conv::shapeInput(t, 1, conv::Padding::zeros);
	o(t);
	
	slice = t.getSlice2d(1);
	std::cout << slice << std::endl;
	slice = t.getSlice2d(2);
	std::cout << slice << std::endl;
	

	std::vector<dimmType> dim_input = {3, 3, 1};
	std::vector<int> input_values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	n_at::Dimension d2(dim_input);
	n_at::Tensor<int> t2(input_values, d2);
	
	o(t2);
	std::vector<int> k_val = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	n_at::Tensor<int> k(k_val, d2);
	
	o(k);
	n_at::Tensor<int> out;
	
	conv::conv2d<int>(out, t2, k, 1, 1, conv::Padding::zeros, 1);

	o(out);

	k.stackSlice2d(out);
	o(k);	
}
