#include <ctime>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define PROCEDURE 1

//-------------------------------------------------------

static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage);

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