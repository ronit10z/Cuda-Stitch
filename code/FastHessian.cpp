#include <vector>
#include <assert.h>

#include "FastHessian.hpp"
#include "ResponseMapGpuGenerator.cu_incl"


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include "Timer.hpp"
extern TimeAccumulator timeAccumulator;

using namespace std;

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32

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


void FastHessian::setGpuIntegralImage(float* integralImage)
{
  this->gpuIntegralImage = integralImage;
}


inline float BoxIntegral(Mat img, int row, int col, int rows, int cols) 
{
  // The subtraction by one for row/col is because row/col is inclusive.
  int r1 = std::min(row,          img.rows) - 1;
  int c1 = std::min(col,          img.cols)  - 1;
  int r2 = std::min(row + rows,   img.rows) - 1;
  int c2 = std::min(col + cols,   img.cols)  - 1;

  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
  if (r1 >= 0 && c1 >= 0) A = img.at<float>(r1, c1);
  if (r1 >= 0 && c2 >= 0) B = img.at<float>(r1, c2);
  if (r2 >= 0 && c1 >= 0) C = img.at<float>(r2, c1);
  if (r2 >= 0 && c2 >= 0) D = img.at<float>(r2, c2);

  return std::max(0.f, A - B - C + D);
}

inline float BoxSumIntegral(const Mat &img, int row, int col, int rows, int cols)
{
  int r1 = std::min(row,          img.rows) - 1;
  int c1 = std::min(col,          img.cols)  - 1;
  int r2 = std::min(row + rows,   img.rows) - 1;
  int c2 = std::min(col + cols,   img.cols)  - 1;

  float sum = 0;
  for (int r = r1; r < r2; ++r)
  {
    for (int c = c1; c < c2; ++c)
    {
      sum += img.at<float>(r, c);
    }
  }
  return sum;

}

//-------------------------------------------------------

//! Constructor with image
FastHessian::FastHessian(Mat &integralImage, Mat &img, std::vector<cv::Point> &ipts, 
                         const int octaves, const int intervals, const int init_sample, 
                         const float thresh) 
                         : integralImage(integralImage), img(img), ipts(ipts)
{
  // Save parameter set
  this->octaves = 
    (octaves > 0 && octaves <= 4 ? octaves : OCTAVES);
  this->intervals = 
    (intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
  this->init_sample = 
    (init_sample > 0 && init_sample <= 6 ? init_sample : INIT_SAMPLE);
  this->thresh = (thresh >= 0 ? thresh : THRES);

  i_height = integralImage.rows;
  i_width = integralImage.cols;


    // Deallocate memory and clear any existing response layers
  for(unsigned int i = 0; i < responseMap.size(); ++i)  
    delete responseMap[i];
  responseMap.clear();

  // Get image attributes
  int w = (i_width / init_sample);
  int h = (i_height / init_sample);
  int s = (init_sample);

  // Calculate approximated determinant of hessian values
  if (octaves >= 1)
  {
    responseMap.push_back(new ResponseLayer(w,   h,   s,   9));
    responseMap.push_back(new ResponseLayer(w,   h,   s,   15));
    responseMap.push_back(new ResponseLayer(w,   h,   s,   21));
    responseMap.push_back(new ResponseLayer(w,   h,   s,   27));
  }
 
  if (octaves >= 2)
  {
    responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 39));
    responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 51));
  }

  if (octaves >= 3)
  {
    responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 75));
    responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 99));
  }

  if (octaves >= 4)
  {
    responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 147));
    responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 195));
  }

  if (octaves >= 5)
  {
    responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 291));
    responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 387));
  }

  uint64_t numIntervalSlots = this->intervals + ((this->octaves - 1) * this->intervals / 2);
  this->gpuDeterminantSize = numIntervalSlots * i_width * i_height * sizeof(float);
  gpuErrchk(cudaMalloc((void**)&(this->gpuDeterminants), this->gpuDeterminantSize));
  gpuErrchk(cudaMemset(this->gpuDeterminants, 0, this->gpuDeterminantSize));

  gpuErrchk(cudaMalloc((void**) &lobeSizesPrecomputed__CUDA, 10*sizeof(int)));
  gpuErrchk(cudaMemcpy(lobeSizesPrecomputed__CUDA, lobeSizesPrecomputed, 10 * sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**) &cudaLobeMap, 16*sizeof(int)));
  gpuErrchk(cudaMemcpy(cudaLobeMap, lobeSizesPrecomputedMap, 16 * sizeof(int), cudaMemcpyHostToDevice));

  this->cudaInterestPointsLen = i_width * i_height / IMG_SIZE_DIVISOR;
  gpuErrchk(cudaMalloc((void**) &cudaInterestPoints, sizeof(cudaPoint) * cudaInterestPointsLen));
  this->hostInterestPoints = (cudaPoint*)malloc(i_width * i_height * sizeof(cudaPoint) / IMG_SIZE_DIVISOR);

  gpuErrchk(cudaMalloc((void**) &atomicCounter, sizeof(int)));


  gpuErrchk(cudaDeviceSynchronize());
}


void FastHessian::SetImage(Mat &integralImage, Mat &img)
{
  this->integralImage = integralImage;
  this->img = img;
  i_height = integralImage.rows;
  i_width = integralImage.cols;
}

//-------------------------------------------------------

FastHessian::~FastHessian()
{
  for (unsigned int i = 0; i < responseMap.size(); ++i)
  {
    delete responseMap[i];
  }
}


//-------------------------------------------------------

//! Find the image features and write into vector of features
void FastHessian::getIpoints()
{
  // filter index map
  static const int filter_map [OCTAVES][INTERVALS] = {{0,1,2,3}, {1,3,4,5}, {3,5,6,7}, {5,7,8,9}, {7,9,10,11}};

  // Clear the vector of exisiting ipts
  ipts.clear();

  // Build the response map
  StartTimer(&timeAccumulator, DET_CAL);
  buildResponseMap();
  EndTimer(&timeAccumulator, DET_CAL);

  // Get the response layers
  ResponseLayer *b, *m, *t;
  StartTimer(&timeAccumulator, NMS);
  for (int o = 0; o < octaves; ++o) for (int i = 0; i <= 1; ++i)
  {
    b = responseMap.at(filter_map[o][i]);
    m = responseMap.at(filter_map[o][i+1]);
    t = responseMap.at(filter_map[o][i+2]);

    // loop over middle response layer at density of the most 
    // sparse layer (always top), to find maxima across scale and space
    for (int r = 0; r < t->height; ++r)
    {
      for (int c = 0; c < t->width; ++c)
      {
        if (isExtremum(r, c, t, m, b))
        {
          Point ipt;
          ipt.x = static_cast<float>(c * t->step);
          ipt.y = static_cast<float>(r * t->step);
          ipts.push_back(ipt);
        }
      }
    }
  }
  EndTimer(&timeAccumulator, NMS);
}

//-------------------------------------------------------

//! Build map of DoH responses
void FastHessian::buildResponseMap()
{
  // Extract responses from the image
  for (unsigned int i = 0; i < responseMap.size(); ++i)
  {
    buildResponseLayer(responseMap[i], i);
  }
}

//-------------------------------------------------------

//! Calculate DoH responses for supplied layer
void FastHessian::buildResponseLayer(ResponseLayer *rl, int responseIndex)
{
  float *responses = rl->responses;         // response storage

  int step = rl->step;                      // step size for this filter
  int b = (rl->filter - 1) / 2 + 1;         // border for this filter
  int l = rl->filter / 3;                   // lobe for this filter (filter size / 3)
  int w = rl->filter;                       // filter size
  float inverse_area = 1.f/(w*w);           // normalisation factor
  float Dxx, Dyy, Dxy;

  int index = 0;
  for(int ar = 0, index = 0; ar < rl->height; ar++) 
  {
    int r = 0;
    int c = 0;
    for(int ac = 0; ac < rl->width; ac++) 
    {
      // get the image coordinates
      r = ar * step;
      c = ac * step; 

      // Compute response components
      Dxx = BoxIntegral(integralImage, r - l + 1, c - b, 2*l - 1, w)
          - BoxIntegral(integralImage, r - l + 1, c - l / 2, 2*l - 1, l)*3;
      Dyy = BoxIntegral(integralImage, r - b, c - l + 1, w, 2*l - 1)
          - BoxIntegral(integralImage, r - l / 2, c - l + 1, l, 2*l - 1)*3;
      Dxy = + BoxIntegral(integralImage, r - l, c + 1, l, l)
            + BoxIntegral(integralImage, r + 1, c - l, l, l)
            - BoxIntegral(integralImage, r - l, c - l, l, l)
            - BoxIntegral(integralImage, r + 1, c + 1, l, l);

      // Normalise the filter responses with respect to their size
      Dxx *= inverse_area;
      Dyy *= inverse_area;
      Dxy *= inverse_area;
     
      responses[index++] = (Dxx * Dyy - 0.81f * Dxy * Dxy);
    }
  }
}

void FastHessian::buildResponseLayer__CUDA()
{

  int currentStep = init_sample;
  for (int i = 0; i < this->octaves; ++i)
  {
    int borderOffset = borderSizesPrecomputed[i];
    // dim3 gridDimensions;
    unsigned int numFilterApplications_x = (this->integralImage.cols + currentStep - 1) / currentStep;
    unsigned int numFilterApplications_y = (this->integralImage.rows + currentStep - 1) / currentStep;  
    unsigned int gridDimension_x = (numFilterApplications_x + BLOCKDIM_X - 1) / BLOCKDIM_X;
    unsigned int gridDimension_y = (numFilterApplications_y + BLOCKDIM_Y - 1) / BLOCKDIM_Y;
    dim3 gridDimensions(gridDimension_x, gridDimension_y);
    dim3 blockDimensions(BLOCKDIM_X, BLOCKDIM_Y);
   
    int actualIntervals = (i == 0) ? 4 : 2;
    gridDimensions.x *= actualIntervals;

    LaunchKernel(gridDimensions, blockDimensions, lobeSizesPrecomputed__CUDA, this->gpuIntegralImage, this->gpuDeterminants, this->integralImage.cols, 
      this->integralImage.rows, actualIntervals, i, currentStep, borderOffset);

    currentStep *= 2;
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

void FastHessian::NMS__CUDA()
{
  int currentStep = init_sample;

  gpuErrchk(cudaMemset(this->atomicCounter, 0, sizeof(int)));
  gpuErrchk(cudaMemset(this->cudaInterestPoints, -1, sizeof(cudaPoint) * this->cudaInterestPointsLen));

  cudaDeviceSynchronize();

  for (int i = 0; i < this->octaves; ++i)
  {
    int extrudedStep = 3 * currentStep;
    int numNMSApplications_x = (this->integralImage.cols + extrudedStep - 1) / (extrudedStep);
    int numNMSApplications_y = (this->integralImage.rows + extrudedStep - 1) / (extrudedStep);
    int intervalsPerNMS = 3;
    int numNMSApplications_intervals = (this->intervals + intervalsPerNMS - 1) / intervalsPerNMS;

    unsigned int gridDimension_x = (numNMSApplications_x + BLOCKDIM_X - 1) / BLOCKDIM_X;
    unsigned int gridDimension_y = (numNMSApplications_y + BLOCKDIM_Y - 1) / BLOCKDIM_Y;
    gridDimension_x *= numNMSApplications_intervals;

    dim3 gridDimensions(gridDimension_x, gridDimension_y);
    dim3 blockDimensions(BLOCKDIM_X, BLOCKDIM_Y);

    LaunchNMSKernel(gridDimensions, blockDimensions, this->gpuDeterminants, this->cudaInterestPoints, this->atomicCounter, 
      this->cudaLobeMap, this->integralImage.cols, this->integralImage.rows, this->intervals, i, 
      currentStep, numNMSApplications_intervals, this->thresh);

    currentStep *= 2;
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(this->hostInterestPoints, this->cudaInterestPoints, sizeof(cudaPoint) * this->cudaInterestPointsLen, cudaMemcpyDeviceToHost);
  
  gpuErrchk(cudaDeviceSynchronize());

  ipts.resize(0);

  for (int i = 0; i < this->cudaInterestPointsLen; ++i)
  {
    int x = hostInterestPoints[i].x;
    int y = hostInterestPoints[i].y;
    if (x == -1 || y == -1) continue;
    ipts.push_back(Point(x, y));
  }
}

  
// -------------------------------------------------------

//! Non Maximal Suppression function
int FastHessian::isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, 
  ResponseLayer *b)
{
  // bounds check
  int layerBorder = (t->filter + 1) / (2 * t->step);
  if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
    return 0;

  // check the candidate point in the middle layer is above thresh 
  float candidate = m->getResponse(r, c, t);
  if (candidate < thresh) 
    return 0; 

  for (int rr = -1; rr <=1; ++rr)
  {
    for (int cc = -1; cc <=1; ++cc)
    {
      // if any response in 3x3x3 is greater candidate not maximum
      if (
        t->getResponse(r+rr, c+cc) >= candidate ||
        ((rr != 0 || cc != 0) && m->getResponse(r+rr, c+cc, t) >= candidate) ||
        b->getResponse(r+rr, c+cc, t) >= candidate
        ) 
        return 0;
    }
  }

  return 1;
}