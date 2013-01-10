// Homework 2
// Gaussian Blur
//
//...

#include "../student/reference_calc.cpp"
#include "../student/utils.h"

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows, int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{

  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  uchar4 rgba = inputImageRGBA[thread_1D_pos];

  redChannel[thread_1D_pos]   = rgba.x;
  greenChannel[thread_1D_pos] = rgba.y;
  blueChannel[thread_1D_pos]  = rgba.z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows, int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}


//Naive Gaussian blur
//Uses only global memory which means there is much
//more global memory traffic than needed
//Could use constant memory for the filter and
//shared memory for the image

//Each thread is mapped to one pixel in the output channel
//It loops over the size of the filter and loads each corresponding
//channel value from global memory.  Then it loads the corresponding
//filter value also from global memory.
__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  float result = 0.f;

  for (int r = -filterWidth/2; r <= filterWidth/2; ++r) {
    for (int c = -filterWidth/2; c <= filterWidth/2; ++c) {
      //the min & max clamp the values so stay within the image
      int image_row = min(max(thread_2D_pos.y + r, 0), numRows - 1);
      int image_col = min(max(thread_2D_pos.x + c, 0), numCols - 1);

      unsigned char channelVal = inputChannel[image_row * numCols + image_col];

      float filterVal = filter[(r + filterWidth/2) * filterWidth + c + filterWidth/2];

      result += filterVal * static_cast<float>(channelVal);
    }
  }

  outputChannel[thread_1D_pos] = static_cast<unsigned char>(result);
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        const float* const h_filter, const int filterWidth)
{
  //allocate memory for the three different channels
  //original and blurred

  unsigned char *d_red, *d_green, *d_blue;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
  float         *d_filter;

  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRows * numCols));

  checkCudaErrors(cudaMalloc(&d_redBlurred,   sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_greenBlurred, sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_blueBlurred,  sizeof(unsigned char) * numRows * numCols));

  //allocate memory for the filter and copy it to the gpu
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

  //written this way to emphasize that the x dimension of the block should be a
  //multiple of the warpSize for coalescing purposes
  const int warpSize = 32;
  const dim3 blockSize(warpSize, 16, 1);
  const dim3 gridSize( (numCols + blockSize.x - 1) / blockSize.x, 
                       (numRows + blockSize.y - 1) / blockSize.y, 1);

  //first kernel to split RGBA into separate channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  checkCudaErrors(cudaDeviceSynchronize());

  //second phase does 3 convolutions, one on each color channel
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols,
                                         d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols,
                                         d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols,
                                         d_filter, filterWidth);
  checkCudaErrors(cudaDeviceSynchronize());

  //last phase recombines
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
                                             d_outputImageRGBA, numRows, numCols);
  checkCudaErrors(cudaDeviceSynchronize());

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

  /*uchar4 *h_outputImage     = new uchar4[numRows * numCols];
  uchar4 *h_outputReference = new uchar4[numRows * numCols];

  checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImageRGBA, 
                             numRows * numCols * sizeof(uchar4), 
                             cudaMemcpyDeviceToHost));

  referenceCalculation(h_inputImageRGBA, h_outputReference, numRows, numCols,
                       h_filter, filterWidth);

  //the 4 is because there are 4 channels in the image
  checkResultsExact((unsigned char *)h_outputReference, (unsigned char *)h_outputImage, numRows * numCols * 4); 
 
  delete [] h_outputImage;
  delete [] h_outputReference;*/

  cudaFree(d_red);
  cudaFree(d_redBlurred);
  cudaFree(d_green);
  cudaFree(d_greenBlurred);
  cudaFree(d_blue);
  cudaFree(d_blueBlurred);
  cudaFree(d_filter);
}
