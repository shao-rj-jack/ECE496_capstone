/*
Author: ss
Date: Nov 12
Description: Tensor and related class definitions for starting tensor vector
//Dimensions stored in col first order -- easier to work with padding
//Tensor stored as row first
*/

#pragma once
#include <vector>
#include <iostream>
#include <cassert>

typedef unsigned long dimmType;

namespace n_at {class Dimension;}
static std::ostream& operator<<(std::ostream& os, const n_at::Dimension& d);

namespace n_at {
	
	class Dimension {
		public:
			Dimension(){}
			//Dimension(dimmType n) : dimm_vector(), size(n){}
			Dimension(std::vector<dimmType> v) : dimm_vector(v), size(v.size()){
				assert(dimm_vector.size() == size);
			}
			Dimension(const Dimension& d) : dimm_vector(), size(){
				*this = d;
			}
			~Dimension(){}
			
			//dont want to allow this
			//you must initialize the dimension, cannot change after
			/*bool push_dimm(dimmType d){
				if(this->getsize() < this->size){
					dimm_vector.push_back(d);
					return true;
				}else return false;
			}*/
			
			void updateVector(const std::vector<dimmType>& v){
				this->dimm_vector = v;
				this->size = v.size();
			}
			
			dimmType  getsize() const{
				return size;
			}
			
			std::vector<dimmType> getVector(){
				return this->dimm_vector;
			}
			
			dimmType getNthDim(dimmType dim_num) const{
				assert(dim_num > 0);
				if(dim_num <= size)
					return dimm_vector[dim_num-1];
				else
					return 1; //3x3 = 3x3x1 
			}
			
			void setNthDim(dimmType num_dim, dimmType dim){
				assert(num_dim <= size);
				
				dimm_vector.at(num_dim -1) = dim;
			   	
			}
			dimmType getNumElems() const{
				dimmType total = 1;
				for(auto i : dimm_vector){
					total *= i;
				}
				return total;
			}
			
			friend std::ostream& ::operator<<(std::ostream& os, const Dimension& d);
		
			Dimension& operator=(const Dimension& rhs){
				if(this == &rhs) return *this;
				
				this->dimm_vector = rhs.dimm_vector;
				this->size = rhs.size;
				return *this;
			}
		private:
			std::vector<dimmType> dimm_vector;
			dimmType size;
	};
} 


static std::ostream& operator<<(std::ostream& os, const n_at::Dimension& d){
	os << "(size=" << d.getsize() << "," << "elems=" << d.getNumElems() << ")\n";
	
	for(dimmType i : d.dimm_vector){
		os << i << ",";
	}
	return os;
}

namespace n_at {template<class T> class Tensor;}

template<class T>
std::ostream& operator<<(std::ostream& os, const n_at::Tensor<T>& t);


//TODO: should make Dimension class virtual and inherit from Dimesnion in Tensor, so that I don't have to provide so many wrapper functions, and all the copy overhead due to it
//TODO:for performance should make vector_ public or make conv functions friend to Tensor class
namespace n_at {
	
	template<class T>
	class Tensor {
		public:
			Tensor(){}
			//Tensor(size_t n, size_t k) : vector_(), dimm_(k), num_elems(n){}
			//Tensor(n_at::Dimension& d) : vector_(), dimm_(d), num_elems(d.getNumElems()){}
			Tensor(const Tensor& t) : vector_(), dimm_(t.dimm_), num_elems(){				
				sliceCount_1d = 0;
				*this = t;
			}			
			Tensor(std::vector<T> v, n_at::Dimension& d) : vector_(v), dimm_(d), num_elems(d.getNumElems()){
				assert(num_elems == vector_.size());
				sliceCount_1d = 0;
			}
			~Tensor(){}
			
			//get slice 1d and keep count , so next call to this function will be the next slice
			//return iterator pairs to start and end of slice  
			auto getSlice1d_next(std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator>& in) {
				in.first = vector_.begin() + sliceCount_1d*dimm_.getNthDim(1);
				in.second = in.first + dimm_.getNthDim(1);
				sliceCount_1d++;	
				return in;
			}
			
			//Get 2d slice given the value of 3rd dimension
			//if slice num > 3rd dimenion, will push into highe dimension
			void stackSlice2d(Tensor<T>& slice2d){
				//tales 2d slice ands to end
				//assumes always adds to 1st 3d slice (don't care about upper dims)
				assert(getDimsize() == 3); //assumes u have explicitly set 3rd dimensino of this vector to be 1;
				assert(getNthDim(1) == slice2d.getNthDim(1));
				assert(getNthDim(2) == slice2d.getNthDim(2));
				
				dimm_.setNthDim(3, dimm_.getNthDim(3) + 1);
				this->num_elems+=slice2d.num_elems;
				
				slice2d.vector_.resize(slice2d.num_elems); //incase vector grew
				assert(slice2d.getDimsize() == 2);
				insertVector(vector_.end(), slice2d.vector_);
			}
			Tensor getSlice2d(dimmType slice_num){
				assert(slice_num > 0);
				dimmType num_row = dimm_.getNthDim(2);
				dimmType num_col = dimm_.getNthDim(1);
				
				dimmType start_ = (slice_num-1)*num_row*num_col;
				dimmType end_ = slice_num*num_row*num_col;
				
				typename std::vector<T>::const_iterator first = vector_.begin() + start_;
				typename std::vector<T>::const_iterator last = vector_.begin() + end_;
				std::vector<dimmType> vdSlice = {num_col, num_row};
				std::vector<T> vSlice(first, last);
				n_at::Dimension dSlice(vdSlice);
				
				n_at::Tensor<T> Tslice(vSlice, dSlice);
				
				return Tslice;
			}
			
			n_at::Dimension getDimension(){
				return this->dimm_;
			}
			
			std::vector<dimmType> getDimVector(){
				return this->dimm_.getVector();
			}
			
			dimmType getNthDim(dimmType dim_num) const {
				return dimm_.getNthDim(dim_num);
			}
			
			void setNthDim(dimmType num_dim, dimmType dim){
				dimm_.setNthDim(num_dim, dim);
			}
			
			dimmType getDimsize() const {
				return dimm_.getsize();
			}
			
			dimmType getNumElems(){
				return num_elems;
			}
			T get_atPos(dimmType R, dimmType C, dimmType D){
				dimmType pos = (D-1)*dimm_.getNthDim(1)*dimm_.getNthDim(2) + (R-1)*dimm_.getNthDim(1) + C;
				
				return vector_[pos-1];
			}	
			
			void set_atPos(dimmType R, dimmType C, dimmType D, T val){
				dimmType pos = (D-1)*dimm_.getNthDim(1)*dimm_.getNthDim(2) + (R-1)*dimm_.getNthDim(1) + C;
				vector_.insert(vector_.begin() + pos-1, val);
			}
			typename std::vector<T>::iterator getVectorIt_index(dimmType i){
				return vector_.begin() + i;
			}
			void updateDimension(std::vector<dimmType>& v){
			//	dimmType orginal_highest = dimm_.getNthDim(dimm_.getsize());
				dimm_.updateVector(v);
				num_elems = dimm_.getNumElems();
			//	dimmType size_lower = 1;
			//	for(dimmType i = 1; i < dimm_.getsize(); i++)
			//		size_lower*=dimm_.getNthDim(i);
				
				//dimmType num_resize = num_elems- size_lower*orginal_highest;
				vector_.resize(num_elems); //do not use reserve - segfaul
			}
			
			//*Insert always creates new
			//does not replace
			void insertVector(typename std::vector<T>::const_iterator start, std::vector<T>& v){
				vector_.insert(start, v.begin(), v.end());	
			}
			void insertVector(typename std::vector<T>::const_iterator start, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end){
				vector_.insert(start, begin, end);	
			}
			
			
			void copyVector(typename std::vector<T>::iterator start, std::vector<T>& v){
				std::copy(v.begin(), v.end(), start);
			}
			
			void copyVector(typename std::vector<T>::iterator start, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end){
				std::copy(begin, end, start);
			}
			
			void printVec() const {
				std::cout << "end-begin " << vector_.end() - vector_.begin() << std::endl;
				std::cout << "size " << vector_.size() << std::endl;
				std::cout << "capacity " << vector_.capacity() << std::endl;
				for( auto i : vector_){
					std::cout << i << ",";
				}
				std::cout << std::endl;
			}
			Tensor& operator=(const Tensor& rhs){
				if(this == &rhs) return *this;
				
				this->vector_ = rhs.vector_;
				this->dimm_ = rhs.dimm_;
				this->num_elems = rhs.num_elems;
				return *this;
			}

			friend std::ostream& ::operator<< <T>(std::ostream& os, const Tensor<T>& t);

		private:
			std::vector<T> vector_;
			n_at::Dimension dimm_;
			dimmType num_elems;
			dimmType sliceCount_1d;
	};	
}

template <class T>
std::ostream& operator<<(std::ostream& os, const n_at::Tensor<T>& t){
	os << t.dimm_ << "\n";
	t.printVec();
	return os;
}


