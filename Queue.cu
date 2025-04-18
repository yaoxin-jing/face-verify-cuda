/*
 *
 *  Created on: 24 Nov 2019
 *  Author: Lina Maudlej
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include "Queue.h" 
using std::cout;
using std::endl;
using std::cerr;
using std::endl;
using std::cerr;

template<class T>
Queue<T>::Queue(){
	qsize=SLOTSNUM;	
}

template<class T>
void Queue<T>::verify_index(int index){
	if(index>=qsize || index<0 ){
		error("Bad index");
		return;
	}
	if(producer_idx==consumer_idx){
		uint64_t * ptr_status=reinterpret_cast<uint64_t*>(arr+(producer_idx));
		if(*ptr_status==1){	
			return;
		}
	}
	if(producer_idx>consumer_idx && index>=consumer_idx && index<producer_idx){
		return;
	}
	if(producer_idx<consumer_idx && (index>= consumer_idx || index<producer_idx )){
		return;
	}
	error("Bad index");
}

template<class T>
void Queue<T>::verify_index_producer_consumer(int index){
	if(index>=qsize || index<0 ){
		error("Bad index");
		return;
	}
}

template<class T>
bool Queue<T>::verify_to_enqueue(){
	uint64_t * ptr_status=reinterpret_cast<uint64_t*>(arr+(producer_idx));
	while(*ptr_status==1); //loop while the producer_idxer say it is ready
	return true;
}

template<class T>
bool Queue<T>::verify_to_dequeue(){
	uint64_t * ptr_status=reinterpret_cast<uint64_t*>(arr+(consumer_idx));
	while(*ptr_status==0); //loop while the producer_idxer say it is not ready
	return true;
}

template<class T>
/*
 *base_addr - base address of the queue (VA),qsize number of requests in there queue (slotsnum) 
 */
Queue<T>::Queue(T* base_addr,int qsize,uint64_t _producer_idx,uint64_t _consumer_idx, QUEUE_TYPE _qtype){//,size_t _HEADER_SIZE,size_t _MSG_SIZE) {
	this->arr = base_addr;
	this->qsize = qsize;
	this->consumer_idx=_consumer_idx;
	this->producer_idx=_producer_idx;
	qtype=_qtype;
	if(qtype==CONSUMER){
		this->consumer_idx=0;
	}else if(qtype==PRODUCER){
		this->producer_idx=0;
	}
}

template<class T>
Queue<T>::Queue(const Queue& q) {
	this->qsize = q.qsize;
	this->consumer_idx=consumer_idx;
	this->producer_idx=producer_idx;
	this->arr = q.arr;
}

template<class T>
Queue<T>::Queue(Queue& q) {
	this->qsize = q.qsize;
	this->consumer_idx=consumer_idx;
	this->producer_idx=producer_idx;
	this->arr = q.arr;
}

template<class T>
Queue<T>::~Queue() {
	//delete[] arr; the allocated memory is not freed here (by host).
}

template<class T>
int Queue<T>::length(){
	int tmp1=(this->producer_idx + 1) % SLOTSNUM;
	int tmp2=(this->consumer_idx + 1) % SLOTSNUM;
	__sync_synchronize();

	if( tmp1== consumer_idx) {
		return qsize;
	}else if(tmp2==producer_idx){
		return 0;
	}else if(producer_idx>consumer_idx){
		return producer_idx-consumer_idx-1; //it is the first empty availble idx to write to
	}else{ //producer_idx<consumer_idx
		return  qsize-(qsize-producer_idx);
	}

	/*
	   if(producer_idx==consumer_idx){
	   uint64_t * ptr_status=reinterpret_cast<uint64_t*>(arr+(producer_idx));
	   if(*ptr_status==0){
	   return 0;
	   }else{
	   return qsize;
	   }
	   }else if(producer_idx>consumer_idx){
	   return producer_idx-consumer_idx; //it is the first empty availble idx to write to
	   }else{ //producer_idx<consumer_idx
	   return qsize-consumer_idx+producer_idx;
	   }*/
}

template<class T>
int Queue<T>::size(){
	//number of request in the queue (MEM_SIZE)/sizeof(Request)
	return this->qsize;
}

template<class T>
Queue<T>& Queue<T>::operator=(const Queue<T>& q) {
	if (this == &q) {
		return *this;
	}
	this->qsize = q.qsize;
	this->arr=q.arr;
	this->consumer_idx=consumer_idx;
	this->producer_idx=producer_idx;
	return *this;
}

template<class T>
const T& Queue<T>::operator[](int index) const {
	verify_index(index);
	return arr[index];
}

template<class T>
const T& Queue<T>::operator[](int index) {
	verify_index(index);
	return arr[index];
}


//Queue pairs for each kerenl invokation with BLOCKSNUM of queue pairs each has its SLOTSNUM.
//We use BLOCKSNUM because we  have QP per TB.
///*****************************************************************
// ******************************************************************/
queue_pairs::queue_pairs(	Request*   send_base_addr,
		int     send_qsize,
		uint64_t*  send_producer_qnex, 
		uint64_t*  send_consumer_qnex,
		QUEUE_TYPE send_qtype,
		Response*  recv_base_addr, //TODO response
		int     recv_qsize,
		uint64_t*  recv_producer_qnex,
		uint64_t*  recv_consumer_qnex,
		QUEUE_TYPE recv_qtype){
	/*  
	CUDA_CHECK(cudaHostAlloc(&send_q, BLOCKSNUM * sizeof(Queue<Request>),
				cudaHostAllocMapped));
	CUDA_CHECK(cudaHostAlloc(&recv_q, BLOCKSNUM * sizeof(Queue<Response>),
				cudaHostAllocMapped));
	//Passes back the device pointer corresponding to the mapped, pinned host buffer allocated by cudaHostAlloc().
	CUDA_CHECK(cudaHostGetDevicePointer(&send_q_device, send_q, 0));
	CUDA_CHECK(cudaHostGetDevicePointer(&recv_q_device, recv_q, 0));
	//send_q=new Queue<Request>((Request*)send_base_addr,send_qsize,send_producer_qnex,send_consumer_qnex,send_qtype);
	//recv_q=new Queue<Request>((Response*)recv_base_addr,recv_qsize,recv_producer_qnex,recv_consumer_qnex,recv_qtype);
	__sync_synchronize();
	CUDA_CHECK(cudaHostAlloc(&flag,  sizeof( bool), cudaHostAllocMapped));
	*flag=false;
	CUDA_CHECK(cudaHostGetDevicePointer(&flag_device, flag, 0));
	__sync_synchronize();
	for (int i = 0; i < BLOCKSNUM; i++) {
		send_queue_init(&send_q[i],SLOTSNUM,send_qtype);
		recv_queue_init(&recv_q[i],SLOTSNUM,recv_qtype);
	}
	reqNum=(int*) malloc(sizeof(int)*BLOCKSNUM);
	respNum=(int*)malloc(sizeof(int)*BLOCKSNUM);
	for(int k=0;k<BLOCKSNUM;k++){
		reqNum[k]=0;
		respNum[k]=0;
	}
	__sync_synchronize();
	*/
};

void queue_pairs::send_queue_init(Queue<Request>* q,int q_size, QUEUE_TYPE q_type) {
	q->producer_idx = 0; //strating form 0
	q->consumer_idx = SLOTSNUM - 1; //starting from 9
	q->qsize=q_size;
	q->qtype=q_type;
}

void queue_pairs::recv_queue_init(Queue<Response>* q,int q_size, QUEUE_TYPE q_type) {
	q->producer_idx = 0; //strating form 0
	q->consumer_idx = SLOTSNUM - 1; //starting from 9
	q->qsize=q_size;
	q->qtype=q_type;
}

void queue_pairs::end_connection(){
	*flag=true;
}

queue_pairs::~queue_pairs(){
	//CUDA_CHECK(cudaFree()); 
	//CUDA_CHECK(cudaFree()); 
	delete reqNum;
	delete respNum;
}
