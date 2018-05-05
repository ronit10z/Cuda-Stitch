#pragma once

#include <opencv/cv.h>
#include "ResponseLayer.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include "ResponseMapGpuGenerator.cu_incl"

using namespace cv;

#include <vector>

static const int OCTAVES = 5;
static const int INTERVALS = 4;
static const float THRES = 1000.0f;
static const int INIT_SAMPLE = 2;

static const int lobeSizesPrecomputed[] = {3, 5, 7, 9, 13, 17, 25, 33, 49, 65};
static const int borderSizesPrecomputed[] = {14, 26, 50, 98};
static const int lobeSizesPrecomputedMap[] = {0, 1, 2, 3, 1, 3, 4, 5, 3, 5, 6, 7, 5, 7, 8, 9};

class FastHessian {
  
  public:
   
	FastHessian(Mat &integralImage, Mat &img, 
		std::vector<cv::Point> &ipts, const int octaves, 
		const int intervals, const int init_sample, const float thresh);

	~FastHessian();

	void SetImage(Mat &integralImage, Mat &img);
	void getIpoints();
	void buildResponseLayer__CUDA();
	void NMS__CUDA();
	void setGpuIntegralImage(float* integralImage);
	void getIpoints__CUDA();
	void asyncCopyIntegralImage__CUDA();
	
	float* gpuIntegralImage;
	uint64_t gpuDeterminantSize;
	float* gpuDeterminants;
	std::vector<ResponseLayer *> responseMap;
	int* lobeSizesPrecomputed__CUDA;
	cudaPoint* hostInterestPoints;
	int cudaInterestPointsLen;

  private:

	//---------------- Private Functions -----------------//

	//! Build map of DoH responses
	void buildResponseMap();

	//! Calculate DoH responses for supplied layer
	void buildResponseLayer(ResponseLayer *r, int responseIndex);

	//! 3x3x3 Extrema test
	int isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);    
	
	//! Interpolation functions - adapted from Lowe's SIFT implementation
	void interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	void interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,
						  double* xi, double* xr, double* xc );
	CvMat* deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	CvMat* hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);

	//---------------- Private Variables -----------------//

	Mat &integralImage;
	Mat &img;
	int i_width, i_height;

	std::vector<cv::Point> &ipts;


	int octaves;

	int intervals;

	int init_sample;

	float thresh;
	int* atomicCounter;
	cudaPoint* cudaInterestPoints;
	int* cudaLobeMap;
	int cudaDeviceId;
};