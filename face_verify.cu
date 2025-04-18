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

__device__ bool isResponseInserted(Queue<Response>* response_queue, Response newResponse) {
	int tmp=(response_queue->producer_idx + 1) % SLOTSNUM;
	__threadfence_system();
	if (tmp == response_queue->consumer_idx) {
		return false;
	}
	response_queue->arr[response_queue->producer_idx] = newResponse;
	__threadfence_system();
	response_queue->producer_idx = tmp;
	__threadfence_system();
	return true;
}

__device__ bool isRequestExists(Queue<Request>* request_queue, Request* req) {
	int tmp=(request_queue->consumer_idx + 1) % SLOTSNUM;
	__threadfence_system();
	if (tmp == request_queue->producer_idx) {
		return false;
	}
	*req = request_queue->arr[tmp];
	__threadfence_system();
	request_queue->consumer_idx = tmp;
	__threadfence_system();
	return true;
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
__global__ void face_recognition(Queue<Request>* request_queue, Queue<Response>* response_queue,bool* flag) {
	Queue<Request>* RequestQueue = &request_queue[blockIdx.x];
	Queue<Response>* ResponseQueue = &response_queue[blockIdx.x];
	__shared__ Request requestDequeued; //output (dequeue response from queue)
	__shared__ int hist1[256];
	__shared__ int hist2[256];
	__shared__ double distance[256];
	while (*flag==false) { //stop all threadblocks from running (queues are emptry)
		if (threadIdx.x == 0) {//when threadnumber is zero try te dequeue a request
			while (!isRequestExists(RequestQueue, &requestDequeued));
			for (int i = 0; i < 256; i++) {
				hist1[i] = 0;
				hist2[i] = 0;
			}
		}
		__syncthreads();
		gpu_image_to_histogram_tmp(requestDequeued.images1, hist1);
		gpu_image_to_histogram_tmp(requestDequeued.images2, hist2);
		__syncthreads();
		gpu_histogram_distance_tmp(hist1, hist2, distance);

		/*unsigned int usec = 2000;
		  long long int cycles_per_usec = 875;
		  long long int curr;
		  long long int start = clock64();
		  long long int total_cycles = usec * cycles_per_usec;
		  while(total_cycles >  0) {
		  curr = clock64();
		  total_cycles -= (curr - start);
		  start = curr;
		  }*/

		__syncthreads(); //making sure all threads are here
		if (threadIdx.x == 0) {
			Response currResponse;
			currResponse.distance = distance[0];
			currResponse.num = requestDequeued.num;
			//currResponse.images1 = requestDequeued.images1;
			//currResponse.images2 = requestDequeued.images2;
			while (!isResponseInserted(ResponseQueue, currResponse)); ////response is ready, enqueue it
		}
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

//it is for one SINGLE TB
/*__global__ void face_verfication_batch_TB(uchar* imges,int* idxes,double* total_dis) {
	__shared__ int hist[batch*256];
	__shared__ double distance[batch*256];
	if (threadIdx.x == 0) {	 	
		for (int i = 0; i < batch*256; i++) {
			hist[i] = 0;
		}
	}
	__syncthreads();
	for(int i=0; i<batch; i++){
		gpu_image_to_histogram_tmp(&imges[i*IMG_DIMENSION * IMG_DIMENSION], &hist[256*i]);
		__syncthreads();
		gpu_histogram_distance_tmp(&hist[256*i], &hist_dataset[256*idxes[i]], &distance[256*i]);
	}
	__syncthreads(); 		
	if (threadIdx.x == 0) {
		for(int i=0; i<batch;i++){	
			total_dis[i]=distance[256*i];
		}
	}
}*/

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

/*__global__ void copy_wait_kernel(Queue<Request>* request_queue, Queue<Response>* response_queue,bool* flag,unsigned int* useconds) {
  Queue<Request>* RequestQueue = &request_queue[blockIdx.x];
  Queue<Response>* ResponseQueue = &response_queue[blockIdx.x];
  __shared__ int hist1[256];
  __shared__ int hist2[256];
  if (threadIdx.x == 0) {//when threadnumber is zero try te dequeue a request
  while (!isRequestExists(RequestQueue, &requestDequeued));
  for (int i = 0; i < 256; i++) {
  hist1[i] = 0;
  hist2[i] = 0;
  }
  }
  __syncthreads();

  __syncthreads(); //making sure all threads are here
  if (threadIdx.x == 0) {
  Response currResponse;
  currResponse.distance=hist1[0];
  currResponse.num = requestDequeued.num;
  while (!isResponseInserted(ResponseQueue, currResponse)); ////response is ready, enqueue it

  }

  }*/


///perforamce func
///////////////////////////////////////////////////////////////////////////
////////////////////////// NUM OF BLOCK ///////////////////////////////////
__device__ int CalBlocksNum(int _threadsUsed){
	/*printf("The usage of three types of resources affect the number of concurrently scheduled blocks: Threads Per Block, Registers Per Block and Shared Memory Per Block. for example: when you start increasing Shared Memory Per Block, the Max Blocks per Multiprocessor will continue to be your limiting factor until you reach a threshold at which Shared Memory Per Block becomes the limiting factor.\n");
	 */
	/*cudaDeviceProp prop;
	  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	// checking how much we have used
	struct cudaFuncAttributes func;
	CUDA_CHECK(cudaFuncGetAttributes(&func,server_kernel));
	// Registers Per Block
	int maxRegisters=prop.regsPerMultiprocessor;
	int regsUsed=func.numRegs;
	int num2=maxRegisters/regsUsed;
	// Shared Memory Per Block
	int maxSharedMem=prop.sharedMemPerMultiprocessor;
	int sharedMemUsed=func.sharedSizeBytes;
	int num3=maxSharedMem/sharedMemUsed;
	//Threads Per Block
	//if(_threadsUsed<prop.maxThreadsPerBlock){
	int maxThreads=prop.maxThreadsPerMultiProcessor;
	int threadsUsed=prop.maxThreadsPerBlock; //not all of them used!! TODO
	int num1=maxThreads/threadsUsed;
	//}
	//printf("max Threads is %d max Registers %d max shared memory %d (per block).\n",num1,num2,num3);
	int num=min(min(num1,num2),num3); // per multiprocessor
	int BLOCKNUM=prop.multiProcessorCount*num;
	//printf("minum between them is: %d, then the numberBlocks: %d.\n",num,BLOCKNUM);

	//printf("block numbers is %d.\n",BLOCKNUM);
	return BLOCKNUM;*/
	return 13; 
} 
