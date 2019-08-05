//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const int BIT_PER_PASS = 4;
const int MAX_BIT_COUNT = 32;
const int GLOBAL_BLOCK_SIZE = 128;

__device__ int getBinId(unsigned int val, int iter, int nBits)
{
	return ((1 << nBits) - 1) & (val >> (iter * nBits));
}

__global__ void sortAndCreateHist(unsigned int * const inpVal, unsigned int * const inpPos, unsigned int * const hists, int nElem, int nBins, int iter, int nBits)
{
	extern __shared__ unsigned int sValue[];
	unsigned int * sPos = sValue + blockDim.x;
	unsigned int * sBit = sValue + blockDim.x + blockDim.x;
	unsigned int * sBinStart = sBit; // Reuse memory for the create histograms part
	unsigned int * sBinId = sPos; // Reuse memory for the create histograms part

	// Load the value of each element to shared memory
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nElem)
	{
		sValue[threadIdx.x] = inpVal[id];
		sPos[threadIdx.x] = inpPos[id];
	}
	__syncthreads();

	// Sort in each block
	// Sort per bit
	for (int k = 0; k < nBits; ++k)
	{
		int b = 0;
		int bPrev = 0;
		
		if (id < nElem)
			b = getBinId(sValue[threadIdx.x], iter * nBits + k, 1);
		// Load previous element to do exclusive scan
		if (threadIdx.x > 0 && id <= nElem)
			bPrev = getBinId(sValue[threadIdx.x - 1], iter * nBits + k, 1); 

		// Save the current bit to shared memory
		sBit[threadIdx.x] = bPrev;
		__syncthreads();

		// Inclusive scan the bit array to get the sum
		// Do the reduction
		int stride = 1;
		for (; stride < blockDim.x; stride *= 2)
		{
			int sid = 2 * stride * (threadIdx.x + 1) - 1;
			if (sid < blockDim.x)
				sBit[sid] += sBit[sid - stride];

			__syncthreads();
		}

		// Do the post-reduction
		for (stride /= 2; stride > 0; stride /= 2)
		{
			int sid = 2 * stride * (threadIdx.x + 1) - 1 + stride;
			if (sid < blockDim.x)
				sBit[sid] += sBit[sid - stride];

			__syncthreads();
		}

		// Calculate the number of zero bit in the bit array
		__shared__ unsigned int numberOfZero;
		if (blockIdx.x < gridDim.x - 1 && threadIdx.x == blockDim.x - 1)
			numberOfZero = blockDim.x - (sBit[threadIdx.x] + b);
		else if (id == nElem - 1) // Special case: last block
			numberOfZero = (threadIdx.x + 1) - (sBit[threadIdx.x] + b);

		__syncthreads();

		// Put to the correct position
		int pos = 0;
		if (b == 0)
			pos = threadIdx.x - sBit[threadIdx.x];
		else
			pos = numberOfZero + sBit[threadIdx.x];

		unsigned int tmpVal = sValue[threadIdx.x];
		unsigned int tmpPos = sPos[threadIdx.x];
		__syncthreads();

		if (id < nElem)
		{
			sValue[pos] = tmpVal;
			sPos[pos] = tmpPos;
		}
		__syncthreads();
	}
	// Write back the sorted results to global memory
	if (id < nElem)
	{
		inpVal[id] = sValue[threadIdx.x];
		inpPos[id] = sPos[threadIdx.x];
	}

	// Find the start position of each bin
	int bid = 0;
	if (id < nElem)
	{
		bid = getBinId(sValue[threadIdx.x], iter, nBits);
		sBinId[threadIdx.x] = bid;
	}
	__syncthreads();

	if (id < nElem)
	{
		if (threadIdx.x == 0)
			sBinStart[bid] = 0;
		else if (bid != sBinId[threadIdx.x - 1])
			sBinStart[bid] = threadIdx.x;
	}
	__syncthreads();

	// Create histogram
	if (id < nElem)
	{
		if (threadIdx.x == blockDim.x - 1 || (bid != sBinId[threadIdx.x + 1]) || id == nElem - 1)
			hists[bid * gridDim.x + blockIdx.x] = threadIdx.x - sBinStart[bid] + 1;
	}
}

void sortBlockAndCreateHist(unsigned int * const d_tmpInputVals, unsigned int * const  d_tmpInputPos, unsigned int * const d_localHists, int nElem, int nBins, int iter, int nBits)
{
	const int BLOCK_SIZE = GLOBAL_BLOCK_SIZE;

	int nBlocks = 1 + (nElem - 1) / BLOCK_SIZE;
	int sharedMemSize = sizeof(unsigned int) * (BLOCK_SIZE * 2 + max(BLOCK_SIZE, nBins));
	sortAndCreateHist<<<nBlocks, BLOCK_SIZE, sharedMemSize>>>(d_tmpInputVals, d_tmpInputPos, d_localHists, nElem, nBins, iter, nBits);
}

__global__ void scanInBlock(unsigned int * inp,	unsigned int * out, unsigned int * bsum, int n)
{
	extern __shared__ unsigned int sMem[];
	// Compute index
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Each block loads 1 element to shared memory
	if (id < n)
		sMem[threadIdx.x] = inp[id];

	__syncthreads();
	// Do the reduction
	int stride = 1;
	for (; stride < n; stride *= 2)
	{
		int sid = 2 * stride * (threadIdx.x + 1) - 1;
		if (sid < blockDim.x)
			sMem[sid] += sMem[sid - stride];

		__syncthreads();
	}

	// Do the post-reduction
	for (stride /= 2; stride > 0; stride /= 2)
	{
		int sid = 2 * stride * (threadIdx.x + 1) - 1 + stride;
		if (sid < blockDim.x)
			sMem[sid] += sMem[sid - stride];

		__syncthreads();
	}

	// Write results to global memory
	if (id < n)
		out[id] = sMem[threadIdx.x];

	if (threadIdx.x == blockDim.x - 1 || id == n - 1)
		bsum[blockIdx.x] = sMem[threadIdx.x];
}

__global__ void increaseBlock(unsigned int * out, unsigned int * inp, int n, int s)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i = id / blockDim.x;
	if (id < n)
		out[id] += inp[i];
}

void findPrefixSumEachBlock(unsigned int * d_input, unsigned int * d_output, int n, int curSize)
{
	const int BLOCK_SIZE = GLOBAL_BLOCK_SIZE;
	if (n <= 1)
		return;
	
	int nBlocks = (1 + (n - 1) / BLOCK_SIZE);
	int sharedMemSize = sizeof(unsigned int) * BLOCK_SIZE;

	// Save the sum of each block for the next iteration
	unsigned int *d_blockSum;
	cudaMalloc(&d_blockSum, sizeof(unsigned int) * nBlocks);

	// Find the prefix sum for each block
	scanInBlock<<<nBlocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_output, d_blockSum, n);

	// Recursive call
	findPrefixSumEachBlock(d_blockSum, d_blockSum, nBlocks, curSize * BLOCK_SIZE);

	// Add the sum of each block to the next block
	int nIb = (nBlocks - 1) * BLOCK_SIZE;
	increaseBlock <<<1 + (nIb - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_output + BLOCK_SIZE, d_blockSum, nIb, curSize);

	checkCudaErrors(cudaFree(d_blockSum));
}

void findExclusivePrefixSum(unsigned int * const d_input, unsigned int * const d_output, int n)
{
	int BLOCK_SIZE = GLOBAL_BLOCK_SIZE;
	findPrefixSumEachBlock(d_input, d_output + 1, n - 1, BLOCK_SIZE);
}

__global__ void scatter(unsigned int * const inpVal,
	unsigned int * const inpPos,
	unsigned int * const outVal,
	unsigned int * const outPos,
	unsigned int * const hist,
	int nElem, int nBins,
	int iter, int nBits)
{
	extern __shared__ unsigned int sBinStart[];
	unsigned int * sVal = sBinStart + nBins;

	// Load value to shared memory
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = 0;
	if (id < nElem)
	{
		bid = getBinId(inpVal[id], iter, nBits);
		sVal[threadIdx.x] = bid;
	}
	__syncthreads();

	// Id in the histScans matrix
	int hid = bid * gridDim.x + blockIdx.x;

	// Find the rank in the block for each element: the number of elements that are to the left and equal
	// We find the starting position of each bin-id and calculate the rank in block from that info
	if (threadIdx.x == 0)
		sBinStart[bid] = 0;
	else if (bid != sVal[threadIdx.x - 1])
		sBinStart[bid] = threadIdx.x;
	__syncthreads();

	int rankInBlock = threadIdx.x - sBinStart[bid];
	
	// Scatter to the correct position
	if (id < nElem)
	{
		int pos = hist[hid] + rankInBlock;
		outVal[pos] = inpVal[id];
		outPos[pos] = inpPos[id];
	}
}	

void scatterToCorrectPos(unsigned int * const d_inputVals, 
	unsigned int * const d_inputPos, 
	unsigned int * const d_outputVals, 
	unsigned int * const d_outputPos, 
	unsigned int * const d_localHistsScan, 
	int nElem, int nBins,
	int iter, int nBits)
{
	const int BLOCK_SIZE = GLOBAL_BLOCK_SIZE;

	int nBlocks = 1 + (nElem - 1) / BLOCK_SIZE;
	int sharedMemSize = sizeof(unsigned int) * (nBins + BLOCK_SIZE);

	scatter<<<nBlocks, BLOCK_SIZE, sharedMemSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, 
		d_localHistsScan, nElem, nBins, iter, nBits);
}

void your_sort(unsigned int * const d_inputVals, 
	unsigned int * const d_inputPos, 
	unsigned int * const d_outputVals, 
	unsigned int * const d_outputPos, 
	const size_t numElems)
{ 
	int numBins = 1 << BIT_PER_PASS;
	int nBlocks = 1 + (numElems - 1) / GLOBAL_BLOCK_SIZE;

	unsigned int * d_localHists;
	checkCudaErrors(cudaMalloc(&d_localHists, sizeof(unsigned int) * numBins * nBlocks));
	
	unsigned int * d_tmpInputVals = d_inputVals;
	unsigned int * d_tmpInputPos = d_inputPos;
	unsigned int * d_tmpOutputVals = d_outputVals;
	unsigned int * d_tmpOutputPos = d_outputPos;

	int nRuns = 0;
	for (int i = 0; i * BIT_PER_PASS < MAX_BIT_COUNT; ++i, ++nRuns)
	{
		checkCudaErrors(cudaMemset(d_localHists, 0, sizeof(unsigned int) * numBins * nBlocks));

		// Sort in each block according to the current group of bits,
		// find the starting position of each bin and create the local histograms
		sortBlockAndCreateHist(d_tmpInputVals, d_tmpInputPos, d_localHists, numElems, numBins, i, BIT_PER_PASS);

		// Do scan on the histogram to compute inclusive prefix sum
		findExclusivePrefixSum(d_localHists, d_localHists, numBins * nBlocks);

		// Do scatter to put each element to its right place
		scatterToCorrectPos(d_tmpInputVals, d_tmpInputPos, d_tmpOutputVals, d_tmpOutputPos, d_localHists, numElems, numBins, i, BIT_PER_PASS);

		// Ping pong
		std::swap(d_tmpInputVals, d_tmpOutputVals);
		std::swap(d_tmpInputPos, d_tmpOutputPos);
	}

	// Copy esults to the correct pointer
	if (nRuns % 2 == 0)
	{
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaFree(d_localHists));
}

/*

Good job!. Your image matched perfectly to the reference image. 

Your program ran and executed in 16.132544 ms.

*/