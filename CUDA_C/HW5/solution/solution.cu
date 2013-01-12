#include "utils.h"
  
__global__
void simpleHisto(const unsigned int* const vals,
                 unsigned int* const histo,
                 int numVals)
{
  const int blockId = (blockIdx.y * gridDim.x + blockIdx.x);
  const int tid = blockId * blockDim.x + threadIdx.x;
  if (tid >= numVals)
    return;

  unsigned int bin = vals[tid];

  atomicAdd(histo + bin, 1);
}

//we launch a small number of blocks that
//go through the entire memory accumulating
//results in the shared memory
template<int numBlocks, int numThreads>
__global__
void fasterHisto1(const unsigned int* const vals,
                  unsigned int* const histo,
                  const unsigned int numVals,
                  const unsigned int numBins)
{
  extern __shared__ int s_bins[];

  const int tid = threadIdx.x + blockIdx.x * numThreads;

  //zero out smem
  #pragma unroll
  for (int i = threadIdx.x ; i < numBins; i += numThreads) {
    s_bins[i] = 0;
  }

  __syncthreads();

  //loop through vals and accum to shared memory
  #pragma unroll
  for (unsigned int i = tid; i < numVals; i += numBlocks * numThreads) {
    unsigned int bin = vals[i];

    atomicAdd(&s_bins[bin], 1);
  }

  __syncthreads();

  //atomically update global histo
  #pragma unroll
  for (int i = threadIdx.x; i < numBins; i += numThreads) {
    atomicAdd(histo + i, s_bins[i]);
  }
}

void computeHistogram(const unsigned int* d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  const unsigned int numThreads = 192;
  const unsigned int numBlocks = 80;

  //grid needs to be 2D to handle large number of elements

  int side = ceil(sqrt(numElems / (double)numThreads));
  dim3 gridSize(side, side, 1);
  //call kernel
  //simpleHisto<<< gridSize, numThreads>>>(d_vals, d_histo, numElems);
  fasterHisto1<numBlocks, numThreads><<<numBlocks, numThreads, numBins * sizeof(unsigned int)>>>(d_vals, d_histo, numElems, numBins);

  checkCudaErrors(cudaDeviceSynchronize());
}
