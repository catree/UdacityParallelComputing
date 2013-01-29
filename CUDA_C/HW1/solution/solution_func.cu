#include "../student/reference_calc.cpp"
#include "../student/utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  const uchar4 rgba            = rgbaImage[thread_1D_pos];
  const unsigned int intensity = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;

  greyImage[thread_1D_pos] = intensity;
}


//Use 1D indexing to not worry about row length, only total # pixels
__global__
void rgba_to_greyscale_faster(const uint2 *const rgbaImage,
                              uchar2 * const greyImage,
                              int numRows, int numCols)
{
  const int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_1D_pos >= numRows * numCols / 2)
    return;

  uint2 in = rgbaImage[thread_1D_pos];

  uchar2 out;

  int tmp1 = in.x & 0x000000FF;
  int tmp2 = (in.x >> 8) & 0x000000FF;
  int tmp3 = (in.x >> 16) & 0x000000FF;

  out.x = (tmp1 + tmp2 + tmp3) / 3;

  int tmp4 = in.y & 0x000000FF;
  int tmp5 = (in.y >> 8) & 0x000000FF;
  int tmp6 = (in.y >> 16) & 0x000000FF;

  out.y = (tmp4 + tmp5 + tmp6) / 3;

  greyImage[thread_1D_pos] = out;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //written this way to emphasize that the x dimension of the block should be a
  //multiple of the warpSize for coalescing purposes
  const int warpSize = 32;
  const dim3 blockSize(warpSize, 16, 1);
  const dim3 gridSize( (numCols + blockSize.x - 1) / blockSize.x, 
                       (numRows + blockSize.y - 1) / blockSize.y, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  
  //This faster process 2 pixels at a time, in case of an add number of pixels it runs
  //an additional kernel that processes one pixel (faster than an if statement inside
  //main kernel).

  /*size_t numPixels = numRows * numCols;
  const dim3 blockSize(512, 1, 1);
  const dim3 gridSize( ( (numPixels / 2) + blockSize.x - 1) / blockSize.x, 1, 1);
  rgba_to_greyscale_faster<<<gridSize, blockSize>>>((uint2 *)d_rgbaImage, (uchar2  *)d_greyImage, numRows, numCols);
  if (numPixels % 2 == 1) {
    rgba_to_greyscale<<<1, 1>>>(d_rgbaImage + numPixels - 1, d_greyImage + numPixels - 1, 1, 1);
  }
  checkCudaErrors(cudaDeviceSynchronize());*/

  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference image on the host by running the          *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated image back to the host and calls a function that compares the   *
  * the two and will output the first location they differ by too much.       *
  * ************************************************************************* */


  /*unsigned char *h_greyImageGPU = new unsigned char[numRows * numCols];
  unsigned char *h_greyImageRef = new unsigned char[numRows * numCols];

  checkCudaErrors(cudaMemcpy(h_greyImageGPU, d_greyImage, 
                             numRows * numCols * sizeof(unsigned char), 
                             cudaMemcpyDeviceToHost));
  referenceCalculation(h_rgbaImage, h_greyImageRef, numRows, numCols);
  checkResultsEps(h_greyImageRef, h_greyImageGPU, numRows * numCols, 1, .001);
 
  delete [] h_greyImageGPU;
  delete [] h_greyImageRef;*/
}
