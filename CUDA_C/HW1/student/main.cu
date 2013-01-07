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
void postProcess();

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols);

int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string filename;
  if (argc == 1) {
    filename = std::string("cinque_terre_small.jpg");
  }
  else if (argc == 2) {
    filename = std::string(argv[1]);
  }
  else {
    std::cerr << "Usage: ./hw# [optional input file]" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, filename);

  tick();
  //call the students' code
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Processing took: " << tick() << " msecs." << std::endl;

  //check results and output the grey image
  postProcess();

  return 0;
}
