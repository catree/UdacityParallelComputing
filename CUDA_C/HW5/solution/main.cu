#include <cstdlib>
#include <iostream>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include <cstdio>
#include <sys/time.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

#include "reference.cpp"

void computeHistogram(const unsigned int* d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems);

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "You must supply an output file" << std::endl;
    exit(1);
  }

  const unsigned int numBins = 1024;
  const unsigned int numElems = 10000 * numBins;
  const float stddev = 10.f;

  unsigned int *vals = new unsigned int[numElems];
  unsigned int *histo = new unsigned int[numBins];

  timeval tv;
  gettimeofday(&tv, NULL);

  srand(tv.tv_usec);

  //make the mean unpredictable, but close enough to the middle
  //so that timings are unaffected
  unsigned int mean = rand() % 100 + 450;

  //Output mean so that grading can happen with the same inputs
  std::cout << "Mean: " << mean << std::endl;

  thrust::minstd_rand rng;

  //stddev of 10 is fairly concentrated - to get really good performance
  //will have to handle the case of a fair amount of contention
  thrust::random::experimental::normal_distribution<float> normalDist((float)mean, stddev);

  for (int i = 0; i < numElems; ++i) {
    vals[i] = min(max((int)normalDist(rng), 0), numBins - 1);
  }

  unsigned int *d_vals, *d_histo;

  GpuTimer timer;

  checkCudaErrors(cudaMalloc(&d_vals,    sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histo,   sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  checkCudaErrors(cudaMemcpy(d_vals, vals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

  timer.Start();
  computeHistogram(d_vals, d_histo, numBins, numElems);
  timer.Stop();
  std::cout << "Histo kernel took " << timer.Elapsed() << " msec" << std::endl;

  timer.Start();
  //calculate reference
  reference_calculation(vals, histo, numElems, numBins);

  timer.Stop();

  std::cout << "Reference took: " << timer.Elapsed() << " msecs" << std::endl;

  unsigned int *h_gpu = new unsigned int[numBins];

  checkCudaErrors(cudaMemcpy(h_gpu, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  std::ofstream ofs(argv[1], std::ios::out | std::iostream::binary);

  ofs.write(reinterpret_cast<char *>(h_gpu), numBins * sizeof(unsigned int));
  ofs.close();

  for (unsigned int i = 0; i < numBins; ++i) {
    if (h_gpu[i] != histo[i]) {
      std::cerr << "Mistmatch at " << i << " " << histo[i] << " " << h_gpu[i] << std::endl;
    }
  }
  
  delete[] h_gpu;
  delete[] vals;
  delete[] histo;

  cudaFree(d_vals);
  cudaFree(d_histo);

  return 0;
}
