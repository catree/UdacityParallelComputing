//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

size_t numRows();  //return # of rows in the image
size_t numCols();  //return # of cols in the image

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                const std::string& filename);

void postProcess(const std::string& output_file);

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        const float* const h_filter, const int filterWidth);

//include the definitions of the above functions for this homework
#include "HW2.cpp"

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;

  float *h_filter;
  int    filterWidth;

  std::string input_file;
  std::string output_file;
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  }
  else {
    std::cerr << "Usage: ./hw# input_file output_file" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &h_filter, &filterWidth, input_file);

  GpuTimer timer;
  timer.Start();
  //call the students' code
  your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(), h_filter, filterWidth);
  timer.Stop();
  checkCudaErrors(cudaDeviceSynchronize());
  int err = printf("e57__TIMING__f82 Processing took: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the blurred image
  postProcess(output_file);

  return 0;
}
