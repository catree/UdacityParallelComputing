//Udacity HW 6
//Poisson Blending

#include "../student/utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/swap.h>
#include "../student/reference_calc.cpp"

struct splitChannels : thrust::unary_function<uchar4, thrust::tuple<unsigned char, unsigned char, unsigned char> >{
  __host__ __device__
  thrust::tuple<unsigned char, unsigned char, unsigned char> operator()(uchar4 pixel) {
    return thrust::make_tuple(pixel.x, pixel.y, pixel.z);
  }
};

struct combineChannels : thrust::unary_function<thrust::tuple<unsigned char, unsigned char, unsigned char>, uchar4> {
  __host__ __device__
  uchar4 operator()(thrust::tuple<unsigned char, unsigned char, unsigned char> t) {
    return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
  }
};

struct sourceMask : thrust::unary_function<uchar4, unsigned char> {
  __host__ __device__
  unsigned char operator()(uchar4 pixel) {
    if (pixel.x == 255 && pixel.y == 255 && pixel.z == 255)
      return 0;
    else
      return 1;
  }
};

//Kernels are all copied from Mike's IPython Notebook


#define BLOCK_SIZE_Y      8
#define BLOCK_SIZE_X      32
#define MAX_NUM_NEIGHBORS 4

__global__ void compute_strict_interior_and_border(
    unsigned char* d_mask,
    unsigned char* d_strict_interior,
    unsigned char* d_border,
    int            num_pixels_y,
    int            num_pixels_x )
{
  __shared__ int s_neighbors[ BLOCK_SIZE_Y ][ BLOCK_SIZE_X ][ MAX_NUM_NEIGHBORS ];

  int  ny       = num_pixels_y;
  int  nx       = num_pixels_x;
  int2 index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
  int  index_1d = ( nx * index_2d.y ) + index_2d.x;

  if ( index_2d.x < nx && index_2d.y < ny )
  {
    int2 index_2d_right = make_int2( index_2d.x + 1, index_2d.y );
    int2 index_2d_up    = make_int2( index_2d.x,     index_2d.y - 1 );
    int2 index_2d_left  = make_int2( index_2d.x - 1, index_2d.y );
    int2 index_2d_down  = make_int2( index_2d.x,     index_2d.y + 1 );
    int  num_neighbors  = 0;

    if ( index_2d_right.x < nx )
    {
      int index_1d_right = ( nx * index_2d_right.y ) + index_2d_right.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_right;
      num_neighbors++;
    }

    if ( index_2d_up.y >= 0 )
    {
      int index_1d_up = ( nx * index_2d_up.y ) + index_2d_up.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_up;
      num_neighbors++;
    }

    if ( index_2d_left.x >= 0 )
    {
      int index_1d_left = ( nx * index_2d_left.y ) + index_2d_left.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_left;
      num_neighbors++;
    }

    if ( index_2d_down.y < ny )
    {
      int index_1d_down = ( nx * index_2d_down.y ) + index_2d_down.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_down;
      num_neighbors++;
    }

    unsigned char mask = d_mask[ index_1d ];

    if ( mask == 1 )
    {
      bool all_neighbor_masks_set = true;

      for ( int i = 0; i < num_neighbors; i++ )
      {
        unsigned char neighbor_mask = d_mask[ s_neighbors[ threadIdx.y ][ threadIdx.x ][ i ] ];

        if ( neighbor_mask == 0 )
        {
          all_neighbor_masks_set = false;
        }
      }

      if ( all_neighbor_masks_set )
      {
        d_strict_interior[ index_1d ] = 1;
        d_border[ index_1d ]          = 0;
      }
      else
      {
        d_strict_interior[ index_1d ] = 0;
        d_border[ index_1d ]          = 1;
      }
    }
    else
    {
      d_border[ index_1d ]          = 0;
      d_strict_interior[ index_1d ] = 0;
    }
  }
}

__global__ void compute_seamless_clone_iteration(
    unsigned char* d_f_star,
    unsigned char* d_g,
    unsigned char* d_strict_interior,
    unsigned char* d_border,
    float*         d_f_current,
    float*         d_f_next,
    int            num_pixels_y,
    int            num_pixels_x )
{
  __shared__ int s_neighbors[ BLOCK_SIZE_Y ][ BLOCK_SIZE_X ][ MAX_NUM_NEIGHBORS ];

  int  ny       = num_pixels_y;
  int  nx       = num_pixels_x;
  int2 index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
  int  index_1d = ( nx * index_2d.y ) + index_2d.x;

  if ( index_2d.x < nx && index_2d.y < ny )
  {
    int2 index_2d_right = make_int2( index_2d.x + 1, index_2d.y );
    int2 index_2d_up    = make_int2( index_2d.x,     index_2d.y - 1 );
    int2 index_2d_left  = make_int2( index_2d.x - 1, index_2d.y );
    int2 index_2d_down  = make_int2( index_2d.x,     index_2d.y + 1 );
    int  num_neighbors  = 0;

    if ( index_2d_right.x < nx )
    {
      int index_1d_right = ( nx * index_2d_right.y ) + index_2d_right.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_right;
      num_neighbors++;
    }

    if ( index_2d_up.y >= 0 )
    {
      int index_1d_up = ( nx * index_2d_up.y ) + index_2d_up.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_up;
      num_neighbors++;
    }

    if ( index_2d_left.x >= 0 )
    {
      int index_1d_left = ( nx * index_2d_left.y ) + index_2d_left.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_left;
      num_neighbors++;
    }

    if ( index_2d_down.y < ny )
    {
      int index_1d_down = ( nx * index_2d_down.y ) + index_2d_down.x;
      s_neighbors[ threadIdx.y ][ threadIdx.x ][ num_neighbors ] = index_1d_down;
      num_neighbors++;
    }

    unsigned char strict_interior = d_strict_interior[ index_1d ];

    if ( strict_interior == 1 )
    {
      float sum_f_current_strict_interior_neighbors = 0.0f;

      for ( int i = 0; i < num_neighbors; i++ )
      {
        unsigned char neighbor_strict_interior = d_strict_interior[ s_neighbors[ threadIdx.y ][ threadIdx.x ][ i ] ];

        if ( neighbor_strict_interior == 1 )
        {
          sum_f_current_strict_interior_neighbors += d_f_current[ s_neighbors[ threadIdx.y ][ threadIdx.x ][ i ] ];
        }
      }

      float sum_f_star_border = 0.0f;

      for ( int i = 0; i < num_neighbors; i++ )
      {
        unsigned char neighbor_border = d_border[ s_neighbors[ threadIdx.y ][ threadIdx.x ][ i ] ];

        if ( neighbor_border == 1 )
        {
          sum_f_star_border += d_f_star[ s_neighbors[ threadIdx.y ][ threadIdx.x ][ i ] ];
        }
      }

      float sum_g = 0.0f;

      sum_g += 4.0f * d_g[ index_1d ];

      for ( int i = 0; i < num_neighbors; i++ )
      {
        sum_g -= d_g[ s_neighbors[ threadIdx.y ][ threadIdx.x ][ i ] ];
      }

      float f_next_value = (sum_f_current_strict_interior_neighbors + sum_f_star_border + sum_g) / (float)num_neighbors;

      d_f_next[ index_1d ] = min( 255.0f, max( 0.0f, f_next_value ) );
    }
  }
}

__global__ void compute_seamless_clone_composite(
    unsigned char* d_f_star,
    float*         d_f,
    unsigned char* d_strict_interior,
    unsigned char* d_seamless_clone,
    int            num_pixels_y,
    int            num_pixels_x )
{
  int  ny       = num_pixels_y;
  int  nx       = num_pixels_x;
  int2 index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
  int  index_1d = ( nx * index_2d.y ) + index_2d.x;

  if ( index_2d.x < nx && index_2d.y < ny )
  {
    unsigned char strict_interior = d_strict_interior[ index_1d ];

    if ( strict_interior == 1 )
    {
      d_seamless_clone[ index_1d ] = (unsigned char)d_f[ index_1d ];
    }
    else
    {
      d_seamless_clone[ index_1d ] = d_f_star[ index_1d ];
    }
  }
}

void your_blend(const uchar4* const h_sourceImg,
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg,
                uchar4* const h_blendedImg){

  size_t imgSize = numRowsSource * numColsSource;
  
  thrust::device_vector<uchar4> d_sourceImg(h_sourceImg, h_sourceImg + imgSize);
  thrust::device_vector<uchar4> d_destImg(h_destImg, h_destImg + imgSize);

  thrust::device_vector<unsigned char> d_red_source(imgSize);
  thrust::device_vector<unsigned char> d_blue_source(imgSize);
  thrust::device_vector<unsigned char> d_green_source(imgSize);

  thrust::device_vector<unsigned char> d_red_destination(imgSize);
  thrust::device_vector<unsigned char> d_blue_destination(imgSize);
  thrust::device_vector<unsigned char> d_green_destination(imgSize);

  //split the channels into components...
  thrust::transform(d_sourceImg.begin(), d_sourceImg.end(),
                    thrust::make_zip_iterator(thrust::make_tuple(d_red_source.begin(),
                                                                 d_blue_source.begin(),
                                                                 d_green_source.begin())),
                    splitChannels());

  thrust::transform(d_destImg.begin(), d_destImg.end(),
                    thrust::make_zip_iterator(thrust::make_tuple(d_red_destination.begin(),
                                                                 d_blue_destination.begin(),
                                                                 d_green_destination.begin())),
                    splitChannels());

  thrust::device_vector<unsigned char> mask(imgSize);

  thrust::transform(d_sourceImg.begin(), d_sourceImg.end(), mask.begin(),
                    sourceMask());

  thrust::device_vector<unsigned char> strictInteriorRegion(imgSize);
  thrust::device_vector<unsigned char> borderRegion(imgSize);

  //compute the interior and border region
  const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  const dim3 gridSize( (numColsSource + blockSize.x - 1) / blockSize.x,
                       (numRowsSource + blockSize.y - 1) / blockSize.y, 1);

  compute_strict_interior_and_border<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(&mask[0]),
                                                              thrust::raw_pointer_cast(&strictInteriorRegion[0]),
                                                              thrust::raw_pointer_cast(&borderRegion[0]),
                                                              numRowsSource, numColsSource);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //thrust doesn't correctly swap device_vectors (it actually performs copies)
  //so we can't make these device_vectors - need to use raw pointers
  float *blendedImage_Red_1;
  float *blendedImage_Red_2;
  float *blendedImage_Blue_1;
  float *blendedImage_Blue_2;
  float *blendedImage_Green_1;
  float *blendedImage_Green_2;

  checkCudaErrors(cudaMalloc(&blendedImage_Red_1, sizeof(float) * imgSize));
  checkCudaErrors(cudaMalloc(&blendedImage_Red_2, sizeof(float) * imgSize));
  checkCudaErrors(cudaMalloc(&blendedImage_Blue_1, sizeof(float) * imgSize));
  checkCudaErrors(cudaMalloc(&blendedImage_Blue_2, sizeof(float) * imgSize));
  checkCudaErrors(cudaMalloc(&blendedImage_Green_1, sizeof(float) * imgSize));
  checkCudaErrors(cudaMalloc(&blendedImage_Green_2, sizeof(float) * imgSize));

  //set initial conditions to source image, since it is fairly close to the solution
  thrust::copy(d_red_source.begin(), d_red_source.end(), thrust::device_ptr<float>(blendedImage_Red_1));
  thrust::copy(d_red_source.begin(), d_red_source.end(), thrust::device_ptr<float>(blendedImage_Red_2));
  thrust::copy(d_blue_source.begin(), d_blue_source.end(), thrust::device_ptr<float>(blendedImage_Blue_1));
  thrust::copy(d_blue_source.begin(), d_blue_source.end(), thrust::device_ptr<float>(blendedImage_Blue_2));
  thrust::copy(d_green_source.begin(), d_green_source.end(), thrust::device_ptr<float>(blendedImage_Green_1));
  thrust::copy(d_green_source.begin(), d_green_source.end(), thrust::device_ptr<float>(blendedImage_Green_2));

  const size_t numIterations = 800;
  
  for (size_t i = 0; i < numIterations; ++i) {
    compute_seamless_clone_iteration<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(&d_red_destination[0]),
        thrust::raw_pointer_cast(&d_red_source[0]),
        thrust::raw_pointer_cast(&strictInteriorRegion[0]),
        thrust::raw_pointer_cast(&borderRegion[0]),
        thrust::raw_pointer_cast(&blendedImage_Red_1[0]),
        thrust::raw_pointer_cast(&blendedImage_Red_2[0]),
        numRowsSource, numColsSource);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    thrust::swap(blendedImage_Red_1, blendedImage_Red_2);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    compute_seamless_clone_iteration<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(&d_blue_destination[0]),
        thrust::raw_pointer_cast(&d_blue_source[0]),
        thrust::raw_pointer_cast(&strictInteriorRegion[0]),
        thrust::raw_pointer_cast(&borderRegion[0]),
        thrust::raw_pointer_cast(&blendedImage_Blue_1[0]),
        thrust::raw_pointer_cast(&blendedImage_Blue_2[0]),
        numRowsSource, numColsSource);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    thrust::swap(blendedImage_Blue_1, blendedImage_Blue_2);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    compute_seamless_clone_iteration<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(&d_green_destination[0]),
        thrust::raw_pointer_cast(&d_green_source[0]),
        thrust::raw_pointer_cast(&strictInteriorRegion[0]),
        thrust::raw_pointer_cast(&borderRegion[0]),
        thrust::raw_pointer_cast(&blendedImage_Green_1[0]),
        thrust::raw_pointer_cast(&blendedImage_Green_2[0]),
        numRowsSource, numColsSource);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    thrust::swap(blendedImage_Green_1, blendedImage_Green_2);
  }

  //output is in _1

  compute_seamless_clone_composite<<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(&d_red_destination[0]),
      thrust::raw_pointer_cast(&blendedImage_Red_1[0]),
      thrust::raw_pointer_cast(&strictInteriorRegion[0]),
      thrust::raw_pointer_cast(&d_red_destination[0]),
      numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  compute_seamless_clone_composite<<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(&d_blue_destination[0]),
      thrust::raw_pointer_cast(&blendedImage_Blue_1[0]),
      thrust::raw_pointer_cast(&strictInteriorRegion[0]),
      thrust::raw_pointer_cast(&d_blue_destination[0]),
      numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  compute_seamless_clone_composite<<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(&d_green_destination[0]),
      thrust::raw_pointer_cast(&blendedImage_Green_1[0]),
      thrust::raw_pointer_cast(&strictInteriorRegion[0]),
      thrust::raw_pointer_cast(&d_green_destination[0]),
      numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                      d_red_destination.begin(),
                      d_blue_destination.begin(),
                      d_green_destination.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                      d_red_destination.end(),
                      d_blue_destination.end(),
                      d_green_destination.end())),
                    d_destImg.begin(),
                    combineChannels());

  thrust::copy(d_destImg.begin(), d_destImg.end(), h_blendedImg);

  /*uchar4* h_reference = new uchar4[imgSize];
  reference_calc(h_sourceImg, numRowsSource, numColsSource,
                 h_destImg, h_reference);
  memcpy(h_blendedImg, h_reference, sizeof(uchar4) * imgSize);


  //checkResultsExact((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * imgSize);
  delete[] h_reference;*/
   

  checkCudaErrors(cudaFree(blendedImage_Red_1));
  checkCudaErrors(cudaFree(blendedImage_Red_2));
  checkCudaErrors(cudaFree(blendedImage_Blue_1));
  checkCudaErrors(cudaFree(blendedImage_Blue_2));
  checkCudaErrors(cudaFree(blendedImage_Green_1));
  checkCudaErrors(cudaFree(blendedImage_Green_2));
}
