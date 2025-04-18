#include "face_verify.h"
uint64_t my_batch;

__device__ __host__ bool is_in_image_bounds(int i, int j) {
	return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}
__device__         __host__ uchar local_binary_pattern(uchar *image, int i, int j) {
	uchar center = image[i * IMG_DIMENSION + j];
	uchar pattern = 0;
	if (is_in_image_bounds(i - 1, j - 1))
		pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
	if (is_in_image_bounds(i - 1, j))
		pattern |= (image[(i - 1) * IMG_DIMENSION + (j)] >= center) << 6;
	if (is_in_image_bounds(i - 1, j + 1))
		pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
	if (is_in_image_bounds(i, j + 1))
		pattern |= (image[(i) * IMG_DIMENSION + (j + 1)] >= center) << 4;
	if (is_in_image_bounds(i + 1, j + 1))
		pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
	if (is_in_image_bounds(i + 1, j))
		pattern |= (image[(i + 1) * IMG_DIMENSION + (j)] >= center) << 2;
	if (is_in_image_bounds(i + 1, j - 1))
		pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
	if (is_in_image_bounds(i, j - 1))
		pattern |= (image[(i) * IMG_DIMENSION + (j - 1)] >= center) << 0;
	return pattern;
}

__device__ void gpu_image_to_histogram_tmp(uchar *image, int *histogram) {
	for (int i = threadIdx.x; i < IMG_DIMENSION * IMG_DIMENSION; i += blockDim.x){
		uchar pattern = local_binary_pattern(image, i / IMG_DIMENSION,
				i % IMG_DIMENSION);
		atomicAdd(&histogram[pattern], 1);
	}
}


__device__ void gpu_histogram_distance_tmp(int *h1, int *h2, double *distance) {
	int length = 256;
	//	if(tid<256){
	for (int tid = threadIdx.x; tid < length; tid += blockDim.x) {
		distance[tid] = 0;
		if (h1[tid] + h2[tid] != 0) {
			distance[tid] = ((double) SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
		}
	}
	__syncthreads();

	while (length > 1) {
		if (threadIdx.x < length / 2) {
			for (int tid = threadIdx.x; tid < length/2; tid += blockDim.x) {
				distance[tid] = distance[tid] + distance[tid + length / 2];
			}
		}
		length /= 2;
		__syncthreads();
	}
	//	}
}



__global__ void gpu_image_to_histogram(uchar *image, int *histogram) {
	uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION,
			threadIdx.x % IMG_DIMENSION);
	atomicAdd(&histogram[pattern], 1);
}

__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance) {
	int length = 256;
	int tid = threadIdx.x;
	distance[tid] = 0;
	if (h1[tid] + h2[tid] != 0) {
		distance[tid] = ((double) SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
	}
	__syncthreads();

	while (length > 1) {
		if (threadIdx.x < length / 2) {
			distance[tid] = distance[tid] + distance[tid + length / 2];
		}
		length /= 2;
		__syncthreads();
	}
}


extern "C"  __global__ void hist_cal(uchar* gpu_images,int his_idx){
	if (threadIdx.x == 0) { 
		for (int i = 0; i < 256; i++) {
			(hist_dataset+(256*his_idx))[i] = 0;
		}
	}
	__syncthreads();
	gpu_image_to_histogram_tmp(gpu_images, &hist_dataset[256*his_idx]);
}
/********************************face_verfication*************************/

__global__ void face_verfication(uchar* img,int idx,double * total_dis) {
	__shared__ int hist[256];
	__shared__ double distance[256];
	if (threadIdx.x == 0) {	 	//when threadnumber is zero try te dequeue a request
		for (int i = 0; i < 256; i++) {
			hist[i] = 0;
		}
	}
	__syncthreads();
	gpu_image_to_histogram_tmp(img, hist);
	__syncthreads();
	gpu_histogram_distance_tmp(hist, &hist_dataset[256*idx], distance);
	__syncthreads(); 		
	if (threadIdx.x == 0) {
		*total_dis=distance[0];
	}
}

extern "C" 
__global__ void face_verfication_batch(uchar* imges,int* idxes,double* total_dis, uint64_t _batch) {
	__shared__ int hist[256];
	__shared__ double distance[256];
	if (threadIdx.x == 0) {	 	
		for (int i = 0; i <256; i++) {
			hist[i] = 0;
			distance[i]=0;
		}
	}
	__syncthreads();
	int num_imgs_per_block=_batch/gridDim.x;	
	int idx_base=num_imgs_per_block*blockIdx.x;
	if(num_imgs_per_block==0){
		num_imgs_per_block=_batch;
	}
	uchar* images_block_base=imges+(idx_base*IMG_DIMENSION * IMG_DIMENSION);
	int* idxes_block_base=(int*)idxes+idx_base;
	for (int i=0;i <num_imgs_per_block;i++){
		int idx=idxes_block_base[i];
		gpu_image_to_histogram_tmp(images_block_base+(i*IMG_DIMENSION * IMG_DIMENSION), hist);
		__syncthreads();
		gpu_histogram_distance_tmp(hist, hist_dataset+(256*idx), distance);
		__syncthreads();
		if (threadIdx.x == 0) {
			(total_dis+idx_base)[i]=distance[0];
			for (int i = 0; i <256; i++) {
				hist[i] = 0;
				distance[i]=0;
			}
		}
		__syncthreads();
	}
}
