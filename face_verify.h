#ifndef	GPU_H_
#define GPU_H_
/*The algorithm takes as input 2 matrices, each represents a grayscale image of a face.
 It then converts each image into a histogram using a technique called Local Binary Patterns.
 It finally computes the distance between the two histograms as the algorithm’s output.Similar faces are expected to produce smaller distances than less similar faces.(Not required) The full algorithm is described in “Ahonen, Timo, Abdenour Hadid, and Matti Pietikainen. "Face description with local binary patterns: Application to face recognition." IEEE transactions on pattern analysis and machine intelligence 28.12 (2006): 2037-2041.”
 */

#include <stdio.h>
#include "./Queue.h"

/*global memory for histograms*/
//__device__ uchar gpu_images[IMG_DIMENSION * IMG_DIMENSION];
__device__ int hist_dataset[1024*256];

__device__         __host__ uchar local_binary_pattern(uchar *image, int i, int j);
__device__ void gpu_image_to_histogram_tmp(uchar *image, int *histogram);
__device__ void gpu_histogram_distance_tmp(int *h1, int *h2, double *distance);
__global__ void face_recognition(Queue<Request>* req_queue, Queue<Response>* resp_queue,bool* flag);
__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance);
__global__ void gpu_image_to_histogram(uchar *image, int *histogram);
__device__ bool isResponseInserted(Queue<Response>* response_queue, Response newResponse); 
__device__ bool isRequestExists(Queue<Request>* req_queue, Request* req);
__global__ void face_verfication(uchar* img,int index,double* distance);
extern "C" __global__ void face_verfication_batch(uchar* imges,int* idxes,double* total_dis, uint64_t batch);
extern "C" __global__ void hist_cal(uchar* gpu_images,int his_idx);
int CalBlocksNum(int _threadsUsed);

#endif /*GPU_H_*/


