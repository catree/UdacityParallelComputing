//Udacity HW 4
//Sorting

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>

#include "../student/reference_calc.cpp"
#include "../student/utils.h"

struct flags : thrust::unary_function<unsigned int, unsigned int> {
  unsigned int mask;
  unsigned int i;

  __host__ __device__
  flags(unsigned int m) : mask(1 << m), i(m) {}

  __host__ __device__
  unsigned int operator()(unsigned int val) {
    return (val & mask) >> i;
  }
};

struct setScatterLoc : thrust::binary_function<unsigned int, unsigned int, thrust::tuple<unsigned int, unsigned int> > {
  unsigned int offset;

  __host__ __device__
  setScatterLoc(unsigned int o) : offset(o) {}

  //first tuple element is the flag
  //second tuple element is the global id
  __host__ __device__
  unsigned int operator()(unsigned int scanVal, thrust::tuple<unsigned int, unsigned int> t) {
    if (thrust::get<0>(t)) {
      return offset + scanVal;
    }
    else {
      return thrust::get<1>(t) - scanVal;
    }
  }
};

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{

  //Solution 1 (Fast): just use thrust's sorting method
  //Will likely be too fast for many students - would consider doubling time

  /*cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

  thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_outputVals),
                      thrust::device_ptr<unsigned int>(d_outputVals) + numElems,
                      thrust::device_ptr<unsigned int>(d_outputPos));*/

  //Solution 2 (Slower): radix is 1 bit, use splitting
  //May still be too fast, might consider doubling again
  thrust::device_vector<unsigned int> flagScan(numElems);
  thrust::device_vector<unsigned int> scatterLoc(numElems);
  thrust::device_ptr<unsigned int> d_inV(d_inputVals);
  thrust::device_ptr<unsigned int> d_inP(d_inputPos);
  thrust::device_ptr<unsigned int> d_outV(d_outputVals);
  thrust::device_ptr<unsigned int> d_outP(d_outputPos);

  for (int i = 0; i < 32; ++i) {
    thrust::exclusive_scan(thrust::make_transform_iterator(d_inV, flags(i)),
                           thrust::make_transform_iterator(d_inV + numElems, flags(i)),
                           flagScan.begin());

    unsigned int offset = numElems - flagScan.back();
    unsigned int lastElem = d_inV[numElems - 1];
    unsigned int mask = 1 << i;
    if (lastElem & mask)
      offset--;

    //this transforms calculate the correct location to scatter to
    thrust::transform(flagScan.begin(), flagScan.end(),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          thrust::make_transform_iterator(d_inV, flags(i)),
                          thrust::make_counting_iterator(0))),
                      scatterLoc.begin(),
                      setScatterLoc(offset));

    thrust::scatter(d_inV, d_inV + numElems, scatterLoc.begin(), d_outV);
    thrust::scatter(d_inP, d_inP + numElems, scatterLoc.begin(), d_outP);

    std::swap(d_inV, d_outV);
    std::swap(d_inP, d_outP);
  }

  thrust::copy(d_inV, d_inV + numElems, d_outV);
  thrust::copy(d_inP, d_inP + numElems, d_outP);
}
