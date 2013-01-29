// Homework 2

#include "../student/reference_calc.cpp"
#include "../student/utils.h"
#include <thrust/extrema.h>
#include <thrust/scan.h>

/* Copied from Mike's IPython notebook with minor changes
   moved to 1D thread indexing.  Constified some input pointers */

/* Seems silly to optimize here since HW5 is about optimizing histograms */

__global__ void compute_histogram(
    unsigned int* const d_histogram,
    const float* const  d_log_Y,
    float         min_log_Y,
    float         max_log_Y,
    float         log_Y_range,
    int           num_bins,
    int           num_pixels)
{

  const int image_index_1d = blockIdx.x * blockDim.x + threadIdx.x;

  if ( image_index_1d < num_pixels )
  {
    float log_Y     = d_log_Y[ image_index_1d ];
    int   bin_index = min( num_bins - 1, int( ( num_bins * ( log_Y - min_log_Y ) ) / log_Y_range ) );

    atomicAdd( d_histogram + bin_index, 1 );
  }
}

//TODO need "slow" versions of min/max and scan


void your_histogram_and_prefixsum(const float* const d_luminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  thrust::device_ptr<float> d_lum((float *)d_luminance);

  /* The thrust routines are well optimized, but there is little reason to find the min/max
     separately plus thrust has its own overheads.  Good students should still be able to beat
     these routines because of that */
  thrust::device_ptr<float> min_it = thrust::min_element(d_lum, d_lum + numRows * numCols);
  thrust::device_ptr<float> max_it = thrust::max_element(d_lum, d_lum + numRows * numCols);

  min_logLum  = *min_it;
  max_logLum = *max_it;

  float range = max_logLum - min_logLum;

  const int numThreads = 512;

  unsigned int *d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  compute_histogram<<< (numRows * numCols + numThreads - 1) / numThreads, numThreads>>>(
      d_histo, d_luminance, min_logLum, max_logLum, range, numBins, numRows * numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  thrust::device_ptr<unsigned int> histo(d_histo);
  thrust::device_ptr<unsigned int> cdf(d_cdf);

  /* The scan is so small, that most techniques will probably not have significant
     difference in their execution times.  Thrust may actual be fairly slow because
     of its high overhead. */
  thrust::exclusive_scan(histo, histo + numBins, cdf);


  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference cdf on the host by running the            *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated cdf back to the host and calls a function that compares the     *
  * the two and will output the first location they differ.                   *
  * ************************************************************************* */

  /* float *h_logLuminance = new float[numRows * numCols];
  unsigned int *h_cdf   = new unsigned int[numBins];
  unsigned int *h_your_cdf = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_luminance, numCols * numRows * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_your_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins);

  //compare the results of the CDF
  checkResultsExact(h_cdf, h_your_cdf, numBins);
 
  delete[] h_logLuminance;
  delete[] h_cdf; 
  delete[] h_your_cdf; */
}
