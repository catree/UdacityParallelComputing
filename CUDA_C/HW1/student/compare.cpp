#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>

int main(int argc, char **argv) {
  if (!(argc == 3 || argc == 4)) {
    std::cerr << "Usage: ./compare goldImage testImage [epsilon]" << std::endl;
    exit(1);
  }

  cv::Mat gold = cv::imread(argv[1], -1);
  cv::Mat test = cv::imread(argv[2], -1);

  if (gold.empty() || test.empty()) {
    std::cerr << "Inputs couldn't be read! " << argv[1] << " " << argv[2] << std::endl;
    exit(1);
  }

  if (gold.channels() != test.channels()) {
    std::cerr << "Images have different number of channels! " << gold.channels() << " " << test.channels() << std::endl;
    exit(1);
  }

  if (gold.size() != test.size()) {
    std::cerr << "Images have different sizes! [" << gold.rows << ", " << gold.cols << "] ["
              << test.rows << ", " << test.cols << "]" << std::endl;
    exit(1);
  }

  //OK, now we can start comparing values...
  unsigned char *goldPtr = gold.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (argc == 3)
    checkResultsExact(goldPtr, testPtr, gold.rows * gold.cols * gold.channels());
  else {
    double epsilon = atof(argv[3]);
    checkResultsEps(goldPtr, testPtr, gold.rows * gold.cols * gold.channels(), epsilon, epsilon);
  }

  std::cout << "PASS" << std::endl;
  return 0;
}
