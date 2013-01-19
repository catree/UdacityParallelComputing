//Udacity HW 4
//Sorting

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../student/reference_calc.cpp"

void your_sort(unsigned int* const inputVals,
               unsigned int* const inputPos,
               unsigned int* const outputVals,
               unsigned int* const outputPos,
               const size_t numElems)
{
  cudaMemcpy(outputVals, inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  cudaMemcpy(outputPos, inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

  thrust::sort_by_key(thrust::device_ptr<unsigned int>(outputVals),
                      thrust::device_ptr<unsigned int>(outputVals) + numElems,
                      thrust::device_ptr<unsigned int>(outputPos));


  /*
  thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(inputVals),
                                                thrust::device_ptr<unsigned int>(inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(inputPos),
                                               thrust::device_ptr<unsigned int>(inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
                        &h_outputVals[0], &h_outputPos[0],
                        numElems); */
}
