#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check( #val, __FILE__, __LINE__)

template<typename T>
void check(T func, const char* const file, const int line) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

template<typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
  //check that the GPU result matches the CPU result
  for (size_t i = 0; i < numElem; ++i) {
    if (ref[i] != gpu[i]) {
      std::cerr << "Difference at pos " << i << std::endl;
      //the + is magic to convert char to int without messing
      //with other types
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                 "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
  }
}

template<typename T>
void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
  assert(eps1 >= 0 && eps2 >= 0);
  unsigned long long totalDiff = 0;
  for (size_t i = 0; i < numElem; ++i) {
    //subtract smaller from larger in case of unsigned types
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > eps1) {
      std::cerr << "Difference at pos exceeds tolerance" << +i << std::endl;
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
        "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
    totalDiff += diff * diff;
  }
  double norm = sqrt(totalDiff) / (double)numElem;
  if (norm > eps2) {
    std::cerr << "Total L2 norm difference between the two images exceeds " << eps2 << std::endl;
    std::cerr << "L2 norm: " << norm << std::endl;
    exit(1);
  }
}

//Uses the autodesk method of image comparison
//Note the the tolerance here is in PIXELS not a percentage of input pixels
template<typename T>
void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
{

  size_t numBadPixels = 0;
  for (size_t i = 0; i < numElem; ++i) {
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > variance)
      ++numBadPixels;
  }

  if (numBadPixels > tolerance) {
    std::cerr << "Too many bad pixels in the image." << numBadPixels << "/" << tolerance << std::endl;
    exit(1);
  }
}

#endif