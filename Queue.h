/*
 *
 *  Created on: 24 Nov 2019
 *  Author: Lina
 */

#ifndef QUEUE_H_
#define QUEUE_H_
#include "./common.h" 
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
using std::cout;
using std::endl;
using std::cerr;
using std::endl;
using std::cerr;
enum QUEUE_TYPE { PRODUCER,CONSUMER };
typedef unsigned char uchar;

template<class T>
class Queue {
	//request or response. it shoud have uint64_t status element. (first 64 bits)
	//T *arr;
	//internal checking
	void verify_index(int index);
	void verify_index_producer_consumer(int index);
	bool verify_to_dequeue();
	bool verify_to_enqueue();
	void error(const char*str) {
		cerr << "Error: " << str << endl;
		exit(0);
	}
	public:
	QUEUE_TYPE qtype;
	int qsize;	
	/*
	   we have to queues, queue for requests and queue for respons
	   every queue has its own "producer_idx" and "consumer_idx" which indiacts the encoded/decoded place in queue
	   every qeueue has an array for requests or respons 
	   which includes msg requested (img1 & img2 ) the request number (for adding calculated time) to the req
	   also 
	   */
	uint64_t consumer_idx;
	uint64_t producer_idx;
	T arr[SLOTSNUM];
	Queue();
	Queue(T* base_addr,int qsize,uint64_t _producer_idx,uint64_t _consumer_idx, QUEUE_TYPE _qtype);//size_t _HEADER_SIZE,size_t _MSG_SIZE);
	Queue(const Queue& q);
	Queue(Queue& q);
	~Queue();

	T dequeue(){
		assert(qtype==CONSUMER);
		while(isEmpty()); //busy wait when it empty
		int tmp=(consumer_idx + 1) % SLOTSNUM;
		T data =this->arr[tmp];
		consumer_idx = tmp;
		__sync_synchronize();
		//while(!verify_to_dequeue());	
		//uint64_t * ptr_status=reinterpret_cast<uint64_t*>(arr+(consumer_idx));
		//T data = arr[(consumer_idx)];
		//consumer_idx=(consumer_idx+1)%qsize;
		//verify_index_producer_consumer((consumer_idx));
		//*ptr_status=0; //the producer allowed to use it.
		return data;
	}

	void enqueue(T item){
		while(isFull()); //busy wait when it full, in this case producer_idx==consumer_idx and the data is valid
		this->arr[producer_idx] = item;
		__sync_synchronize();
		//cudaStreamQuery waits for other reqs in stream finishes 
		this->producer_idx = (producer_idx + 1) % SLOTSNUM;
		__sync_synchronize();
		//while(!verify_to_enqueue());	 
		//uint64_t * ptr_status=reinterpret_cast<uint64_t*>(arr+(producer_idx));
		//arr[producer_idx]=item;
		//producer_idx=(producer_idx+1)%qsize;
		//verify_index_producer_consumer(producer_idx);
		//*ptr_status=1; //the consumer allowed to use it.
	}
	int length();
	int size();
	bool isEmpty();
	bool isFull();
	Queue& operator=(const Queue& q);
	const T& operator[](int index) const;
	const T& operator[](int index);

};


template<class T>
bool Queue<T>::isEmpty() {
	assert(qsize==SLOTSNUM);
	int tmp=(this->consumer_idx + 1) % SLOTSNUM;
	__sync_synchronize();
	if(tmp==producer_idx){
		return true;
	}
	return false;
	//return (this->length() == 0);
}

template<class T>
bool Queue<T>::isFull() {
	assert(qsize==SLOTSNUM);
	int tmp=(this->producer_idx + 1) % SLOTSNUM;
	__sync_synchronize();
	if( tmp== consumer_idx) {
		return true;
	}
	return false;
	//return (this->length() == qsize);
}

class queue_pairs{
	public:
		Queue<Request>* send_q; //send_q from the server point of view
		Queue<Response>* recv_q;//recv_q from the server point of view
		//the same qp but in the device memory
		Queue<Request>* send_q_device; 
		Queue<Response>* recv_q_device;
		bool* flag;
		bool* flag_device;
		//  respNum & reqNum to indicate if there are requests to dequeue before killing threadblocks
		int* respNum;//reponses counter
		int* reqNum; //requests counter
		queue_pairs(	Request*   send_base_addr,
				int     send_qsize,
				uint64_t*  send_producer_qnex, 
				uint64_t*  send_consumer_qnex,
				QUEUE_TYPE send_qtype,
				Response*  recv_base_addr, //TODO response
				int     recv_qsize,
				uint64_t*  recv_producer_qnex,
				uint64_t*  recv_consumer_qnex,
				QUEUE_TYPE recv_qtype);
		void send_queue_init(Queue<Request>* q,int q_size, QUEUE_TYPE q_type);
		void recv_queue_init(Queue<Response>* q,int q_size, QUEUE_TYPE q_type);
		void end_connection();
		~queue_pairs();
};

#endif /* QUEUE_H_ */

