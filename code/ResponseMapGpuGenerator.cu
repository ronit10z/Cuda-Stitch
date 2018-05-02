#include "ResponseMapGpuGenerator.cu_incl"

__constant__ int lobeSizesPrecomputed__CUDA[10];

void LaunchKernel(dim3 gridDimensions, dim3 blockDimensions, float* gpuIntegralImage, float* determinants, int width ,int height, 
		int numIntervals, int octaveNum, int stepSize, int borderOffset)
{
	GetResponses__CUDA <<<gridDimensions, blockDimensions>>> (gpuIntegralImage, determinants, width, height, numIntervals, octaveNum, stepSize, borderOffset);
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

__global__ void GetResponses__CUDA(float* integralImage, float* determinants, int width ,int height, 
		int numIntervals, int octaveNum, int stepSize, int borderOffset)

{
	int integralImageRow = (blockIdx.y * blockDim.y + threadIdx.y);
	// need to mul by step size to take downsampling into account
	integralImageRow *= stepSize;
	// outside border is all zero. MAGA
	integralImageRow += borderOffset;

	// need to know the size in block that corresponds to one intervals worth 
	int blocksPerInterval_x = gridDim.x / numIntervals;
	int integralImageCol = (blockIdx.x % blocksPerInterval_x);
	integralImageCol *= blockDim.x; // getting down to thread idx
	integralImageCol += threadIdx.x;
	integralImageCol *= stepSize;
	integralImageCol += borderOffset;

	int lobeSizesPrecomputedOffset = blockIdx.x / blocksPerInterval_x;
	// taking advantage of redundant work, octaves after 0 can skip first two integrals
	// but the first one has to compute all 4 :(
	if (octaveNum != 0)
	{
		lobeSizesPrecomputedOffset += 2;
	}
	// filter construction

	// Terminology borrowed from original SURF paper.
	const int lobeSize = lobeSizesPrecomputed__CUDA[octaveNum * numIntervals + lobeSizesPrecomputedOffset];
	const int filterSize = lobeSize * 3;
	const int borderSize = filterSize / 2;

	// Filter responses to be used in approximated determinant.  Taken from original SURF paper.
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

  // Approximation formula taken from original SURF paper
  float determinant = (Dxx * Dyy - 0.81f * Dxy * Dxy);

  //because we compute multiple intervals worth of responses per kernel launch, determine which interval we're currently on
  //value will be between 0 and 3 for octave 0, and either 0 or 1 for octaves > 0
  unsigned int computed_interval = octaveNum * numIntervals + lobeSizesPrecomputedOffset;
  unsigned int interval_start_index = computed_interval * width * height;
  unsigned int determinant_pixel_index = interval_start_index + (integralImageRow * width + integralImageCol);
  determinants[determinant_pixel_index] = determinant;
}