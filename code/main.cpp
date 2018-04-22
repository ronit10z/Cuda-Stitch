#include <ctime>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include "InterestPoint.hpp"
#include "ResponseLayer.hpp"
#include "FastHessian.hpp"

using namespace cv;
using namespace std;

//-------------------------------------------------------

static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage);
void drawIpoints(Mat img, vector<InterestPoint> &ipts);

int main(int argc, char const *argv[])
{
  // Declare Ipoints and other stuff
  Mat img = imread("../../images/bryce_left_02.png");
  Mat gray8(img.size(), CV_8U, 1);
  Mat gray32(img.size(), CV_32F, 1);

  cvtColor(img, gray8, CV_BGR2GRAY);
  gray8.convertTo(gray32, CV_32F, 1.0/255.0, 0);

  Mat integralImage(img.size(), CV_32F, 1);
  GenerateIntegralImage(gray32, integralImage);

  std::vector<InterestPoint> ipts;
  FastHessian fh(integralImage, ipts, 5, 4, 2, 0.0004f);
  fh.getIpoints();

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
  float s, o;
  int r1, c1, r2, c2, lap;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    ipt = &ipts.at(i);
    s = (2.5f * ipt->scale);
    o = ipt->angle;
    lap = ipt->laplaceValue;
    r1 = fRound(ipt->position.second);
    c1 = fRound(ipt->position.first);
    c2 = fRound(s * cos(o)) + c1;
    r2 = fRound(s * sin(o)) + r1;

    if (o) // Green line indicates orientation
      line(img, Point(c1, r1), Point(c2, r2), Scalar(0, 255, 0));
    else  // Green dot if using upright version
      circle(img, Point(c1,r1), 1, cvScalar(0, 255, 0),-1);

    if (lap == 1)
    { // Blue circles indicate dark blobs on light backgrounds
      circle(img, Point(c1,r1), fRound(s), Scalar(255, 0, 0),1);
    }
    else if (lap == 0)
    { // Red circles indicate light blobs on dark backgrounds
      circle(img, Point(c1,r1), fRound(s), Scalar(0, 0, 255),1);
    }
    else if (lap == 9)
    { // Red circles indicate light blobs on dark backgrounds
      circle(img, Point(c1,r1), fRound(s), Scalar(0, 255, 0),1);
    }
  }
}
