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


/* The following is copied directly from Mike's IPython notebook eke */

#define BLOCK_SIZE_Y           8
#define BLOCK_SIZE_X           32
#define BLUR_KERNEL_HALF_WIDTH 4
#define SHARED_MEMORY_SIZE_Y   BLOCK_SIZE_Y + ( 2 * BLUR_KERNEL_HALF_WIDTH )
#define SHARED_MEMORY_SIZE_X   BLOCK_SIZE_X + ( 2 * BLUR_KERNEL_HALF_WIDTH ) + 1
#define SHARED_MEMORY_OFFSET_Y BLUR_KERNEL_HALF_WIDTH
#define SHARED_MEMORY_OFFSET_X BLUR_KERNEL_HALF_WIDTH

__global__ void shared_memory_blur(
    unsigned char* d_blurred,
    unsigned char* d_original,
    float*         d_blur_kernel,
    int            num_pixels_y,
    int            num_pixels_x,
    int            blur_kernel_half_width,
    int            blur_kernel_width )
{
  __shared__ unsigned char s_original[ SHARED_MEMORY_SIZE_Y ][ SHARED_MEMORY_SIZE_X ];

  int  ny                            = num_pixels_y;
  int  nx                            = num_pixels_x;
  int2 image_index_2d_global         = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
  int2 image_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_index_2d_global.x ) ), min( ny - 1, max( 0, image_index_2d_global.y ) ) );
  int  image_index_1d_global_clamped = ( nx * image_index_2d_global_clamped.y ) + image_index_2d_global_clamped.x;
  int2 image_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y );

  //
  // load center of shared memory
  //
  s_original[ image_index_2d_shared_memory.y ][ image_index_2d_shared_memory.x ] = d_original[ image_index_1d_global_clamped ];

  //
  // load y+1 halo into shared memory
  //
  if ( threadIdx.y < BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + 1 ) * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y + BLOCK_SIZE_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load y-1 halo into shared memory
  //
  if ( threadIdx.y >= BLOCK_SIZE_Y - BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( ( blockIdx.y - 1 ) * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y - BLOCK_SIZE_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load x+1 halo into shared memory
  //
  if ( threadIdx.x < BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( ( blockIdx.x + 1 ) * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X + BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load x-1 halo into shared memory
  //
  if ( threadIdx.x >= BLOCK_SIZE_X - BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( ( blockIdx.x - 1 ) * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X - BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load x+1,y+1 halo into shared memory
  //
  if ( threadIdx.x < BLUR_KERNEL_HALF_WIDTH && threadIdx.y < BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( ( blockIdx.x + 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + 1 ) * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X + BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y + BLOCK_SIZE_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load x+1,y-1 halo into shared memory
  //
  if ( threadIdx.x < BLUR_KERNEL_HALF_WIDTH && threadIdx.y >= BLOCK_SIZE_Y - BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( ( blockIdx.x + 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y - 1 ) * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X + BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y - BLOCK_SIZE_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load x-1,y+1 halo into shared memory
  //
  if ( threadIdx.x >= BLOCK_SIZE_X - BLUR_KERNEL_HALF_WIDTH && threadIdx.y < BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( ( blockIdx.x - 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + 1 ) * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X - BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y + BLOCK_SIZE_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // load x-1,y-1 halo into shared memory
  //
  if ( threadIdx.x >= BLOCK_SIZE_X - BLUR_KERNEL_HALF_WIDTH && threadIdx.y >= BLOCK_SIZE_Y - BLUR_KERNEL_HALF_WIDTH )
  {
    int2 image_halo_index_2d_global         = make_int2( ( ( blockIdx.x - 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y - 1 ) * blockDim.y ) + threadIdx.y );
    int2 image_halo_index_2d_global_clamped = make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
    int  image_halo_index_1d_global_clamped = ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
    int2 image_halo_index_2d_shared_memory  = make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X - BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y - BLOCK_SIZE_Y );

    s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
  }

  //
  // wait until all threads in the thread block are finished loading the image chunk into shared memory
  //
  __syncthreads();

  //
  // perform blur operation by reading image from shared memory
  //
  if ( image_index_2d_global.x < nx && image_index_2d_global.y < ny )
  {
    float result = 0;

    for ( int y = -blur_kernel_half_width; y <= blur_kernel_half_width; y++ )
    {
      for ( int x = -blur_kernel_half_width; x <= blur_kernel_half_width; x++ )
      {
        int2          image_offset_index_2d = make_int2( image_index_2d_shared_memory.x + x, image_index_2d_shared_memory.y + y );

        unsigned char image_offset_value    = s_original[ image_offset_index_2d.y ][ image_offset_index_2d.x ];

        int2          blur_kernel_index_2d  = make_int2( x + blur_kernel_half_width, y + blur_kernel_half_width );
        int           blur_kernel_index_1d  = ( blur_kernel_width * blur_kernel_index_2d.y ) + blur_kernel_index_2d.x;

        float         blur_kernel_value     = d_blur_kernel[ blur_kernel_index_1d ];

        result += blur_kernel_value * image_offset_value;
      }
    }

    d_blurred[ image_index_1d_global_clamped ] = (unsigned char)result;
  }
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
  const dim3 blockSize(warpSize, 8, 1);
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

  //Uncomment the lines below for a version that uses shared memory version

  /*shared_memory_blur<<<gridSize, blockSize>>>(d_redBlurred, d_red, d_filter, numRows, numCols, filterWidth/2, filterWidth);
  shared_memory_blur<<<gridSize, blockSize>>>(d_greenBlurred, d_green, d_filter, numRows, numCols, filterWidth/2, filterWidth);
  shared_memory_blur<<<gridSize, blockSize>>>(d_blueBlurred, d_blue, d_filter, numRows, numCols, filterWidth/2, filterWidth);*/
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
