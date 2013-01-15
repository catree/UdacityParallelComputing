/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point
  for each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in through
  stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are less than 3!  This
  means that certain areas (the windows) are extremely bright compared to everywhere else.
  If we linearly map this [0-275] range into the [0-255] range that we have been using
  then most values will be mapped to zero! 
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide range
  of intensity that exists in the real world our monitors are not capable of displaying them.
  Our eyes are also quite capable of observing a much wider range of intensities than
  our image formats / monitors are capable of displaying.

  Tone-mapping is a process that transforms the intensities in the image so that the
  brightest values aren't nearly so far away from the mean.  That way when we transform
  the values into [0-255] we can actually see the entire image.  There are many ways
  to perform this process and it is as much an art as a science - there is no single
  "right" answer.  In this homework we will implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of
  as one possible set of axes spanning a three dimensional space of color.
  We sometimes choose other axes to represent this space because
  they make certain operations more convenient.

  Another possible set of axes is known as Chrominance-Luminance or xyY.
  This set of axes separates the color and intensity information completely -
  that is, the Luminance channel is actually a greyscale version of the image
  and the two Chrominance channels carry the color information.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions
  would display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel
  (actually the log of the luminance, but this is unimportant for the parts of the
   algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an inclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */




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
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, numCols * numRows * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_your_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins);

  //compare the results of the CDF
  checkResultsExact(h_cdf, h_your_cdf, numBins);
 
  delete[] h_logLuminance;
  delete[] h_cdf; 
  delete[] h_your_cdf; */
}
