#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>



__device__ float BoxIntegralCuda(cv::cuda::GpuMat integralImage, int row, int col, int rows, int cols)
{
  int r1 = 	min(row,          integralImage.rows) - 1;
  int c1 = 	min(col,          integralImage.cols)  - 1;
  int r2 = 	min(row + rows,   integralImage.rows) - 1;
  int c2 = 	min(col + cols,   integralImage.cols)  - 1;

  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
  // if (r1 >= 0 && c1 >= 0) A = integralImage.at<float>(r1, c1);
  // if (r1 >= 0 && c2 >= 0) B = integralImage.at<float>(r1, c2);
  // if (r2 >= 0 && c1 >= 0) C = integralImage.at<float>(r2, c1);
  // if (r2 >= 0 && c2 >= 0) D = integralImage.at<float>(r2, c2);

  return max(0.f, A - B - C + D);
}