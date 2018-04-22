#include <ctime>
#include <iostream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include "InterestPoint.hpp"
#include "ResponseLayer.hpp"
#include "FastHessian.hpp"
#include "Timer.hpp"
#include "Brief.hpp"

using namespace cv;
using namespace std;

extern TimeAccumulator timeAccumulator;

//-------------------------------------------------------

static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage);
void drawIpoints(Mat img, vector<InterestPoint> &ipts);

int main(int argc, char const *argv[])
{
  // Declare Ipoints and other stuff
  SetupTimer(&timeAccumulator);
  StartTimer(&timeAccumulator, TOTAL_TIME);

  Mat img = imread("../../images/bryce_left_02.png");
  StartTimer(&timeAccumulator, IMAGE_CONVERSION);
  Mat gray8(img.size(), CV_8U, 1);
  Mat gray32(img.size(), CV_32F, 1);
  cvtColor(img, gray8, CV_BGR2GRAY);
  gray8.convertTo(gray32, CV_32F, 1.0/255.0, 0);
  EndTimer(&timeAccumulator, IMAGE_CONVERSION);

  StartTimer(&timeAccumulator, SUMMED_TABLE);
  Mat integralImage(img.size(), CV_32F, 1);
  GenerateIntegralImage(gray32, integralImage);
  EndTimer(&timeAccumulator, SUMMED_TABLE);

  std::vector<InterestPoint> ipts;
  FastHessian fh(integralImage, ipts, 5, 4, 5, 0.00004f);
  fh.getIpoints();

  EndTimer(&timeAccumulator, TOTAL_TIME);
  PrintTimes(&timeAccumulator);
  printf("Number of Interest Points Detected = %u\n", (uint32_t)ipts.size());

  drawIpoints(img, ipts);

  imshow("l", img);
  waitKey(0);

  return 0;
}



// pass by reference
static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage)
{

  integralImage.at<float>(0, 0) = source.at<float>(0, 0);
  for (int j = 1; j < integralImage.cols; ++j)
  {
    integralImage.at<float>(0, j) = integralImage.at<float>(0, j - 1) + 
    source.at<float>(0, j);
  }

  for (int i = 1; i < integralImage.rows; ++i)
  {
    float accumulator = 0.0f;
    for (int j = 0; j < integralImage.cols; ++j)
    {
      accumulator += source.at<float>(i, j);
      integralImage.at<float>(i, j) = accumulator + integralImage.at<float>(i - 1, j);
    }
  }
}


inline int fRound(float flt)
{
  return (int) floor(flt+0.5f);
}

//! Draw all the Ipoints in the provided vector
void drawIpoints(Mat img, vector<InterestPoint> &ipts)
{
  InterestPoint *ipt;
  int r1, c1;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    ipt = &ipts.at(i);
    r1 = fRound(ipt->position.second);
    c1 = fRound(ipt->position.first);
    // c2 = fRound(s * cos(o)) + c1;
    // r2 = fRound(s * sin(o)) + r1;

    // if (o) // Green line indicates orientation
    //   line(img, Point(c1, r1), Point(c2, r2), Scalar(0, 255, 0));
    // else  // Green dot if using upright version
      circle(img, Point(c1,r1), 1, cvScalar(0, 255, 0),-1);

    // if (lap == 1)
    // { // Blue circles indicate dark blobs on light backgrounds
    //   circle(img, Point(c1,r1), fRound(s), Scalar(255, 0, 0),1);
    // }
    // else if (lap == 0)
    // { // Red circles indicate light blobs on dark backgrounds
    //   circle(img, Point(c1,r1), fRound(s), Scalar(0, 0, 255),1);
    // }
    // else if (lap == 9)
    // { // Red circles indicate light blobs on dark backgrounds
    //   circle(img, Point(c1,r1), fRound(s), Scalar(0, 255, 0),1);
    // }
  }
}
