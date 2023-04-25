#include<stdint.h>
#include<iostream>

int main(){
    int64_t nvtx_scale = ((int64_t)1)<<14;

    uint64_t* cost = (uint64_t*)malloc(sizeof(uint64_t)*nvtx_scale);

    for(int64_t i=0; i < nvtx_scale; i++)
        cost[i] = uint64_t(123456789);

    uint64_t* dcost;
	cudaError_t err;

    err = cudaMalloc(&dcost, nvtx_scale*sizeof(uint64_t));
    if(err!= cudaSuccess) std::cout << "ERROR " << err << std::endl;

	err = cudaMemcpy(dcost, cost, sizeof(uint64_t)*nvtx_scale, cudaMemcpyHostToDevice);
	if(err!= cudaSuccess) std::cout << "ERROR " << err << std::endl;


    memset(cost, 0, sizeof(uint64_t)*nvtx_scale);
    	
	err = cudaMemcpy(cost, dcost, sizeof(uint64_t)*nvtx_scale, cudaMemcpyDeviceToHost);
	
	if(err!= cudaSuccess) std::cout << "ERROR " << err << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << i << " " << cost[i] << std::endl;
    }

    return 0;
}
