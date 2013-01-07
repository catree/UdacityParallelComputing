//Udacity HW1 Solution

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>

size_t numRows();  //return # of rows in the image
size_t numCols();  //return # of cols in the image

//load image into both host and device memory space
//allocate memory for result in both memory spaces
//input = rgbaImage; output = greyImage
//referenceCalculation fills in h_greyImage
//you need to fill in d_greyImage
void preProcess(uchar4 **h_rgbaImage, unsigned char **h_greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string& filename);

//check the results and write out the final image
void postProcess(const std::string& output_file);

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols);

int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

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
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  tick();
  //call the students' code
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "e57__TIMING__f82 Processing took: " << tick() << " msecs." << std::endl;

  //check results and output the grey image
  postProcess(output_file);

  return 0;
}
