#ifndef COMMON_H_
#define COMMON_H_
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib> 
#include <memory> //shared_ptr
#include <map>
#include <vector>
#include <iostream>
#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
#define PAGE_SIZE 2097152
#define RECV_BUF_SIZE (PAGE_SIZE/2)
#define SEND_BUF_SIZE (PAGE_SIZE/2)
#define REQS_COUNT 1
#define N_REGS 1
//lenet 28
#define IMG_DIMENSION 32
//lenet 320

//cliet-server params
#define N_REQUESTS 64 // 1000 is WAMUP
#define BURST N_REQUESTS  //N_REQUESTS
#define MAX_CLIENTS 4
#define WARMUP 0
//QP param
#define SLOTSNUM 10
#define CUDA_CHECK(f) do {                                                                  \
	cudaError_t e = f;                                                                      \
	if (e != cudaSuccess) {                                                                 \
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
		exit(1);                                                                            \
	}                                                                                       \
} while (0)
#define SQR(a) ((a) * (a))

//gpu parameters 
// we use BLOCKSNUM for gpu kernel parameter and for create queue pairs, each TB has its own queue pair.
#define THREADS_QUEUE 1024

enum device_opcode { VCA, GPU, SSD, DEVICES_NUM };
enum req_opcode { CREATE_SLICE,DESTROY_SLICE,SLICE_MEMORY_ALLOCATE,SLICE_REGISTER_FUNCTION,SLICE_CALL_FUNCTION, REQ_NUM};
enum func_opcode { NULL_FUNC, FACE_RECOGNITION, LENET, FACE_VERIFICATION, FUNC_NUM}; //FACE_RECOGNITION hardcoded with 1 , LENET hardcoded with 2

enum mode_opcode{
	NULL_MODE, PROGRAM_MODE_STREAMS , PROGRAM_MODE_QUEUE 
};

enum SERVICE_ERROR{SERVICE_SUCCESS,SERVICE_INVALID_SLICE_ID, SERVICE_INVALID_FUNC_ID,SERVICE_INVALID_SRV_PARAM,SERVICE_ERROR_NUM};

typedef unsigned char uchar;
using namespace std;

double static inline get_time_usec(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6L + t.tv_nsec * 1e-3;
}

//memory
///*****************************************************************
// ******************************************************************/
struct memory{
	char* base_addr; //from shared memory
	size_t size;
};


#endif /*COMMON_H_*/

