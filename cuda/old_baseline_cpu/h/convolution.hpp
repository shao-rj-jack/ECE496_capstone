/*
Author: ss
Date: Nov 12
Description: conv2d basic implementation header file
//only padding more zeros supported so far
Description: conv2d basic implementation, right now channels is not supported, or by default just 1 in/out channels

*/
#pragma once
#include "Tensor.hpp"
#include <cmath>
#define o(x) std::cout << #x << x << std::endl
namespace conv {
	enum Padding {none=0, zeros, reflect, replicate, circular};
	
	
	//out is refrence can be changed, in and kernel are copy will not be altered
	template<class T>
	n_at::Tensor<T>& conv2d(n_at::Tensor<T>&out, n_at::Tensor<T> input, n_at::Tensor<T> kernel, unsigned long stride, unsigned long size_padding, conv::Padding padding, unsigned long dilation);
	
	template<class T>
	n_at::Tensor<T>& shapeKernel(n_at::Tensor<T>& kernel, unsigned long dilation);
	
	template<class T>
	n_at::Tensor<T>& shapeInput(n_at::Tensor<T>& input, unsigned long size_padding, conv::Padding padding);
	
	template<class T>
	void recursivePadd(n_at::Tensor<T>& orig_T, n_at::Tensor<T>& input, std::vector<T>& zero_padd_vector, dimmType dimension, dimmType size_padding, dimmType start_index);
	
	
}

template<class T>
n_at::Tensor<T>& conv::shapeKernel(n_at::Tensor<T>& kernel, unsigned long dilation){
	}

//TODO: currently only support zero padded
template<class T>
n_at::Tensor<T>& conv::shapeInput(n_at::Tensor<T>& input, unsigned long size_padding, conv::Padding padding){
	
	if(input.getDimsize() == 1) return input;
	
	n_at::Tensor<T> orig_T = input;
	
	std::vector<dimmType> df = input.getDimVector();
	for(auto it = df.begin(); it != df.end(); it++){
		*it = *it + 2*(size_padding); //every dimension is changed 
	}
	input.updateDimension(df);
	//reduce to base case which is 1d padding, go from highest dimension
	//to lowest dimension
	dimmType d = input.getDimsize();
		
	dimmType this_DimensionSliceSize = 1;
	dimmType padding_before_after = 1;		
	dimmType interior_padding = 0;
	
	if(d != 1){
		//symetriic padding before = after
		//padding in nth dimension = num_elems of all previous dimensions
		for(dimmType this_d = d; this_d >=1; this_d--){
			this_DimensionSliceSize *= input.getNthDim(this_d);
		}
		padding_before_after = this_DimensionSliceSize/input.getNthDim(d);
		interior_padding = (orig_T.getNthDim(d)) * padding_before_after;
	}
	else{ //d=1
		this_DimensionSliceSize = input.getNthDim(d);
		padding_before_after = size_padding;
		interior_padding = (orig_T.getNthDim(d));
	}
	
		
	//number of interior paddings is equal to this dimension minus padding
	//the size of padded lower d-1 dimensions is equal to padding_before_after b/c padding is applied to all dimensions 	
			
	std::vector<T> zero_padd_vector;
	zero_padd_vector.resize(padding_before_after);
			
	dimmType start = 0;
	dimmType end = start + padding_before_after + interior_padding;
		
	//insert padding at start and end
	//NOTE?? for 2d padding this makes sense
	//BUT FOR 2d convolution do not need 3d padding
	//this is because the convolution does not iterate over the 3rd dimension
	//NOTE** this phenomina ins nsaturally avoided if you define a 2d matrix
	//bc/ final dim is not added (x1) explicity 
	input.copyVector(input.getVectorIt_index(start), zero_padd_vector);
	input.copyVector(input.getVectorIt_index(end), zero_padd_vector);
		
	for(dimmType count = 1; count <= orig_T.getNthDim(d); count++){
		recursivePadd(orig_T, input, zero_padd_vector, d-1, size_padding, count*(start+padding_before_after)); 
	}
	return input;	
}

//TODO: memoize for perf
//recursivly adds zeo padd vector to the correct locations based on padding
//zero padd vector size will always decrease therefore can pas as refrence and resize
//assumes that this is called after top most dimension is dealt with
template<class T>
void conv::recursivePadd(n_at::Tensor<T>& orig_T, n_at::Tensor<T>& input, std::vector<T>& zero_padd_vector, dimmType dimension, dimmType size_padding, dimmType start_index){
	dimmType this_DimensionSliceSize = 1;
	dimmType padding_before_after = 1;		
	dimmType interior_padded = 0;
	if(dimension != 1){
		//symetriic padding before = after
		//padding in nth dimension = num_elems of all previous dimensions
		for(dimmType this_d = dimension; this_d >=1; this_d--){
			this_DimensionSliceSize *= input.getNthDim(this_d);
		}
		padding_before_after = this_DimensionSliceSize/input.getNthDim(dimension);
		interior_padded = (orig_T.getNthDim(dimension)) * padding_before_after;
	}
	else{ //d=1
		this_DimensionSliceSize = input.getNthDim(dimension);
		padding_before_after = size_padding;
		interior_padded = (orig_T.getNthDim(dimension));
	}
	
	zero_padd_vector.resize(padding_before_after);
	input.copyVector(input.getVectorIt_index(start_index), zero_padd_vector);
	input.copyVector(input.getVectorIt_index(start_index + padding_before_after + interior_padded), zero_padd_vector);

	dimmType num_Interior = orig_T.getNthDim(dimension);
	dimension--;

	if(dimension == 0){
		//copy can do in order b/c the copy will be done in order b/c loop over interior dimensions is done inorder
		std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator> copy_v;
		orig_T.getSlice1d_next(copy_v);	
		input.copyVector(input.getVectorIt_index(start_index + padding_before_after), copy_v.first, copy_v.second);	
		return;
	}
	for(dimmType count = 1; count <= num_Interior; count++){
		dimmType start_new = count*this_DimensionSliceSize;
		recursivePadd(orig_T, input, zero_padd_vector, dimension, size_padding, (start_index+count*padding_before_after));
	}
}



//does not deal with outchannels
//outchannels is multiple calls to this function but with different kernels
//wrapper function provides the functionality of pushing a tensor onto another

template<class T>
n_at::Tensor<T>& conv::conv2d(n_at::Tensor<T>&out, n_at::Tensor<T> input, n_at::Tensor<T> kernel, unsigned long stride, unsigned long size_padding, conv::Padding padding, unsigned long dilation){
	
	//NO SUPPORT FOR DILATION //
	
	assert(input.getNthDim(3) == kernel.getNthDim(3));
	
	dimmType input_cols = input.getNthDim(1);
	dimmType input_rows = input.getNthDim(2);
	dimmType input_depth = input.getNthDim(3);
	
	dimmType kernel_cols = kernel.getNthDim(1);
	dimmType kernel_rows = kernel.getNthDim(2);
	
	//right now stride, padding, dilation all symmetric in x/y
	dimmType numC_out = (input.getNthDim(1) + 2*size_padding - dilation*(kernel.getNthDim(1) -1) - 1)/stride + 1;
	dimmType numR_out = (input.getNthDim(2) + 2*size_padding - dilation*(kernel.getNthDim(2) -1) - 1)/stride + 1;
	
	// appropriately size output tensor
	std::vector<dimmType> out_dimm = {numC_out, numR_out};
	
	out.updateDimension(out_dimm);

	if(padding != conv::Padding::none)
		conv::shapeInput(input, size_padding, padding);
	else {/*nothing input  = input*/}
	
	o(input);
		
	for(dimmType i_row = 1 + size_padding; i_row <= 1 + input_rows; i_row+=stride){
		for(dimmType i_col = 1 + size_padding; i_col <= 1 + input_cols; i_col+=stride){
			T sum = 0;
			dimmType i_depth = 1 + size_padding;
			for(; i_depth <= 1 + input_depth; i_depth++){
				
				for(int k_row = -(floor((double)kernel_rows/2)); k_row <= (kernel_rows - ceil((double) kernel_rows/2)); k_row++){
					
					for(int k_col = -((double)floor(kernel_cols/2)); k_col <= (kernel_cols - ceil( (double) kernel_cols/2)); k_col++){
						sum += input.get_atPos(i_row + k_row, i_col + k_col, i_depth) * kernel.get_atPos(k_row + ceil((double)kernel_rows/2), k_col + ceil((double)kernel_cols/2), i_depth-size_padding);
					}
				}
			}
			//will be pin order so can use pushback?
			out.set_atPos(i_row-size_padding, i_col-size_padding, i_depth-size_padding-1, sum);
		}
	}
	
		
	return out;
}

