#include <ctime>
#include <iostream>
#include <chrono>
#include <bitset>

#include "ResponseMapGpuGenerator.cu_incl"



#define CUDA_ERROR_CHECK

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void LaunchKernel(dim3 gridDimensions, dim3 blockDimensions, int* lobeSizesPrecomputed__CUDA, float* gpuIntegralImage, float* determinants, int width ,int height, 
		int numIntervals, int octaveNum, int stepSize, int borderOffset)
{
	GetResponses__CUDA <<<gridDimensions, blockDimensions>>> (lobeSizesPrecomputed__CUDA, gpuIntegralImage, determinants, width, height, numIntervals, octaveNum, stepSize, borderOffset);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}



void LaunchNMSKernel(dim3 gridDimensions, dim3 blockDimensions, float* determinants, cudaPoint* ipts, int* atomicCounter, int* lobeMap, int width, int height, int numIntervals, int octaveNum, 
  int stepSize, int numNMSApplications_intervals, float thresh)
{
   // LaunchNMSKernel(gridDimensions, blockDimensions, this->gpuDeterminants, this->cudaInterestPoints, this->atomicCounter, 
   //    this->cudaLobeMap, this->integralImage.cols, this->integralImage.rows, this->intervals, i, 
   //    currentStep, numNMSApplications_intervals, this->thresh);

  NMS__CUDA<<<gridDimensions, blockDimensions>>>(determinants, ipts, lobeMap, atomicCounter, width, height, numIntervals, octaveNum, stepSize, numNMSApplications_intervals, thresh);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

}


__device__ inline float BoxIntegral__CUDA(float* integralImage, int width, int height, int row, int col, int rows, int cols)
{
  int r1 = 	min(row,          height) - 1;
  int c1 = 	min(col,          width)  - 1;
  int r2 = 	min(row + rows,   height) - 1;
  int c2 = 	min(col + cols,   width)  - 1;

  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
  // width step is in mempry how much each row is(number of bytes in a row, like actually)
  // pretty much the same as in the sequential version, now just need to interface directly with the data struct :(
  if (r1 >= 0 && c1 >= 0) A = integralImage[r1 * width + c1];
  if (r1 >= 0 && c2 >= 0) B = integralImage[r1 * width + c2];
  if (r2 >= 0 && c1 >= 0) C = integralImage[r2 * width + c1];
  if (r2 >= 0 && c2 >= 0) D = integralImage[r2 * width + c2];

  return max(0.f, A - B - C + D);
}

__global__ void GetResponses__CUDA(int* lobeSizesPrecomputed__CUDA, float* integralImage, float* determinants, int width ,int height, 
		int numIntervals, int octaveNum, int stepSize, int borderOffset)

{
	int integralImageRow = (blockIdx.y * blockDim.y + threadIdx.y) * stepSize;

	int blocksPerInterval_x = gridDim.x / numIntervals;
	int integralImageCol = (blockIdx.x % blocksPerInterval_x);
	integralImageCol *= blockDim.x; // getting down to thread idx
	integralImageCol += threadIdx.x;
	integralImageCol *= stepSize;

  if (integralImageCol >= width || integralImageRow >= height) 
  {
    return;
  }

	int lobeSizesPrecomputedOffset = blockIdx.x / blocksPerInterval_x + (octaveNum > 0) * 2;

	const int lobeSize = lobeSizesPrecomputed__CUDA[octaveNum * numIntervals + lobeSizesPrecomputedOffset];;
	const int filterSize = lobeSize * 3;
	const int borderSize = filterSize / 2 + 1;

	float Dxx = BoxIntegral__CUDA(integralImage, width, height, integralImageRow - lobeSize + 1, integralImageCol - borderSize, 2*lobeSize - 1, filterSize)
          - BoxIntegral__CUDA(integralImage, width, height, integralImageRow - lobeSize + 1, integralImageCol - lobeSize / 2, 2*lobeSize - 1, lobeSize)*3;
  float Dyy = BoxIntegral__CUDA(integralImage, width, height, integralImageRow - borderSize, integralImageCol - lobeSize + 1, filterSize, 2*lobeSize - 1)
          - BoxIntegral__CUDA(integralImage, width, height, integralImageRow - lobeSize / 2, integralImageCol - lobeSize + 1, lobeSize, 2*lobeSize - 1)*3;
  float Dxy = + BoxIntegral__CUDA(integralImage, width, height, integralImageRow - lobeSize, integralImageCol + 1, lobeSize, lobeSize)
            + BoxIntegral__CUDA(integralImage, width, height, integralImageRow + 1, integralImageCol - lobeSize, lobeSize, lobeSize)
            - BoxIntegral__CUDA(integralImage, width, height, integralImageRow - lobeSize, integralImageCol - lobeSize, lobeSize, lobeSize)
            - BoxIntegral__CUDA(integralImage, width, height, integralImageRow + 1, integralImageCol + 1, lobeSize, lobeSize);

  float inverseArea = 1.f / (filterSize * filterSize);
  Dxx *= inverseArea;
  Dyy *= inverseArea;
  Dxy *= inverseArea;

  float determinant = (Dxx * Dyy - 0.81f * Dxy * Dxy);

  unsigned int computed_interval = octaveNum * numIntervals + lobeSizesPrecomputedOffset;
  unsigned int interval_start_index = computed_interval * width * height;
  unsigned int determinant_pixel_index = interval_start_index + (integralImageRow * width + integralImageCol);

  // if(octaveNum == 1 && computed_interval == 4) printf("%d %d %d %d\n", determinant_pixel_index - 3 *(width * height), integralImageRow, integralImageCol, computed_interval);
  determinants[determinant_pixel_index] = determinant;
}

__device__ inline static float GetDeterminantFromRelativeOffset(float* determinants, int* lobeMap, int octaveNum, int intervalOffset, int rowOffset, int colOffset, 
  int numIntervals, int width, int height)
{
  int currentInterval = lobeMap[octaveNum * numIntervals + intervalOffset];
  int detOffset = currentInterval * (width * height);
  int detIndex = detOffset + (rowOffset * width) + colOffset;

  return determinants[detIndex];
}

__device__ inline static bool AreNeighborsInBound(int bestRow, int bestCol, int bestInterval, int stepSize, int numIntervals, int width, int height)
{
  //I think this is right, but, is numintervals 4 or 10 here? Which should it be?
  bool intervalInBound = (bestInterval > 0) && (bestInterval < numIntervals - 1);

  //I think these should be >= and not just > as it was before, because if bestRow = stepSize we can subtract a stepSize and get to row 0
  bool rowInBound = (bestRow >= stepSize) && (bestRow < height - stepSize);
  bool colInBound = (bestCol >= stepSize) && (bestCol < width - stepSize);

  return intervalInBound && rowInBound && colInBound;
}

__global__ void NMS__CUDA(float* determinants, cudaPoint* ipts, int* lobeMap, int* atomicCounter, int width, int height, int numIntervals, int octaveNum, 
  int stepSize, int numNMSApplications_intervals, float thresh)
{

   // LaunchNMSKernel(gridDimensions, blockDimensions, this->gpuDeterminants, this->cudaInterestPoints, this->atomicCounter, 
   //    this->cudaLobeMap, this->integralImage.cols, this->integralImage.rows, this->intervals, i, 
   //    currentStep, numNMSApplications_intervals, this->thresh);

  int extrudedStep = 3 * stepSize;
  int blocksPerNMS = gridDim.x / numNMSApplications_intervals;

  int firstNMSBlock = (blockIdx.x / blocksPerNMS) * 3;

  // corresponds to the row and col in one of the points in the nms application, 
  // so it is every 3 beacuse nms is 3x3x3
  int row = (blockIdx.y * blockDim.y + threadIdx.y) * extrudedStep;
  int col = (blockIdx.x % blocksPerNMS);
  col *= blockDim.x;
  col += threadIdx.x;
  col *= 3 * stepSize;

  if (row >= height || col >= width || firstNMSBlock >= numIntervals)
  {
    return;
  }

  float bestVal = -1;
  int bestCol = -1;
  int bestRow = -1;
  int bestInterval = -1;


  //Find the maximum response in a 3x3x3 region across rows, cols, intervals
  for (int intervalOffset = firstNMSBlock; intervalOffset < min(firstNMSBlock + 3, numIntervals - 1); ++intervalOffset)
  {
    for (int rowOffset = row; rowOffset < min(row + extrudedStep, height); rowOffset += stepSize)
    {
      for (int colOffset = col; colOffset < min(col + extrudedStep, width); colOffset += stepSize)
      {
        float currentVal = GetDeterminantFromRelativeOffset(determinants, lobeMap, octaveNum, intervalOffset, rowOffset, colOffset, 
          numIntervals, width, height);

        if (currentVal > bestVal)
        {
          bestVal = currentVal;
          bestCol = colOffset;
          bestRow = rowOffset;
          bestInterval = intervalOffset;
        }
      }
    }
  }

  //Extend the maximal search if there are neighbors to search across
  if (AreNeighborsInBound(bestRow, bestCol, bestInterval, stepSize, numIntervals, width, height))
  {
    //the loopguard is a <= rather than just a <
    for (int i = bestInterval - 1; i <= bestInterval + 1; ++i)
    {
      for (int r = bestRow - stepSize; r <= bestRow + stepSize; r += stepSize)
      {
        for (int c = bestCol - stepSize; c <= bestCol + stepSize; c += stepSize)
        {
          //Get the value in the downsampled image pyramid from the interval offset in the octave, row, and col
          if (i != 0 || r != 0 || c != 0) {
            float currentVal = GetDeterminantFromRelativeOffset(determinants, lobeMap, octaveNum, i, r, c, 
            numIntervals, width, height);
            if (currentVal > bestVal) return;
          }
        }
      }
    }
  }

  // printf("Threshold: %f\n", thresh);

  //if we get to this point, the candiate point is a local extremum
  if (bestVal > thresh && bestInterval != -1)
  {
    int index = atomicAdd(atomicCounter, 1);
    if(index  < width * height)
    {
      ipts[index].x = bestCol;
      ipts[index].y = bestRow;
    }
  }
}