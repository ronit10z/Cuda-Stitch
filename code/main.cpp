#include <ctime>
#include <iostream>
#include <chrono>
#include <bitset>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
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
using namespace cv::xfeatures2d;  
using namespace std;

extern TimeAccumulator timeAccumulator;


#define PATCH_SIZE 9
#define DIM 400
#define SAMPLING_RATE 4

//-------------------------------------------------------

static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage_1);
void drawIpoints(Mat img_1, vector<InterestPoint> &ipts);
// static inline void GetPanSize(double &xMin, double &xMax, double &yMin, double &yMax, Mat &img, Mat &homographyMat);
// Mat getTranslationMatrix(float x, float y);
// void stitch(int height, int width, Mat &img_1, Mat &img_2, 
//   Mat &homographyMat_1, Mat &homographyMat_2);


static inline int CountOnes(uint64_t num)
{
  int count = 0;
  while(num != 0)
  {
    count += (num & 0x1);
    num >>= 1;
  }

  return count;
}

//spit out the descriptor in bit form on cout for debugging if needed
void printBriefDescriptor(BriefPointDescriptor &descript) {
  std::cout << std::bitset<64>(descript.desc_array[3]) << std::bitset<64>(descript.desc_array[2])
    << std::bitset<64>(descript.desc_array[1]) << std::bitset<64>(descript.desc_array[0]) << "\n";
}

Mat createMask(Mat& im) {
    Mat mask = Mat::ones(im.size(), CV_8UC1);
    mask.row(0).setTo(0);
    mask.row(mask.rows-1).setTo(0);
    mask.col(0).setTo(0);
    mask.col(mask.cols-1).setTo(0);
    distanceTransform(mask, mask, CV_DIST_L2, 3);
    double min, max;
    minMaxLoc(im, &min, &max);
    mask /= max;

    std::vector<Mat> singleChannels;
    singleChannels.push_back(mask);
    singleChannels.push_back(mask);
    singleChannels.push_back(mask);
    merge(singleChannels, mask);

    return mask;
}

std::vector<Point2d> getWarpCorners(Mat& im, Mat& H) {
    std::vector<Point2d> im_corners, im_corners_warped;
    im_corners.reserve(4);

    // corners before warping
    int h = im.rows;
    int w = im.cols;
    im_corners.push_back(Point2d(0,0));
    im_corners.push_back(Point2d(w,0));
    im_corners.push_back(Point2d(0,h));
    im_corners.push_back(Point2d(w,h));

    perspectiveTransform(im_corners, im_corners_warped, H);

    return im_corners_warped;
}

Mat getTranslationMatrix(float x, float y) {
    Mat M = Mat::eye(3, 3, CV_32F);
    M.at<float>(0,2) = x;
    M.at<float>(1,2) = y;
    return M;
}

Mat stitchImages(Mat& pano, Mat& image, Mat& H, Mat& pano_mask, Mat& img_mask) 
{
  int width = pano.cols;
  int height = pano.rows;

  Mat image_warped;
  image_warped.create(image.size(), image.type());

  warpPerspective(image, image_warped, H, Size(width, height));

  Mat bim1, bim2;
  img_mask.convertTo(img_mask, image_warped.type());
  multiply(pano, pano_mask, bim1);
  multiply(image_warped, img_mask, bim2);

  Mat stitch_img;
  divide(bim1 + bim2, pano_mask + img_mask, stitch_img);
  return stitch_img;
}


int main(int argc, char const *argv[])
{
  // TODO: change filename to command line arguement, have assert or 
  // exit if argc is not enough.
  // const char* fileName = "../../../../ParaPano/data/testPattern.txt";
  const char* fileName = "../pointOffsets.txt";


  // Declare Ipoints and other stuff
  SetupTimer(&timeAccumulator);
  StartTimer(&timeAccumulator, TOTAL_TIME);

  // // Open image and conver to grey scale
  StartTimer(&timeAccumulator, IMAGE_IO);
  Mat img_1 = imread("../../images/mountainL.png");
  resize(img_1, img_1, Size(DIM, DIM));
  Mat gray8_1(img_1.size(), CV_8U, 1);
  Mat gray32_1(img_1.size(), CV_32F, 1);
  cvtColor(img_1, gray8_1, CV_BGR2GRAY);
  gray8_1.convertTo(gray32_1, CV_32F, 1.0/255.0, 0);
  img_1.convertTo(img_1, CV_32FC3);
  img_1 /= 255.0;

  Mat img_2 = imread("../../images/mountainR.png");
  resize(img_2, img_2, Size(DIM, DIM));
  Mat gray8_2(img_2.size(), CV_8U, 1);
  Mat gray32_2(img_2.size(), CV_32F, 1);
  cvtColor(img_2, gray8_2, CV_BGR2GRAY);
  gray8_2.convertTo(gray32_2, CV_32F, 1.0/255.0, 0);
  img_2.convertTo(img_2, CV_32FC3);
  img_2 /= 255.0;

  EndTimer(&timeAccumulator, IMAGE_IO);

  // // Compute Summed table representation of image
  StartTimer(&timeAccumulator, SUMMED_TABLE);

  Mat integralImage_1(img_1.size(), CV_32F, 1);
  GenerateIntegralImage(gray32_1, integralImage_1);

  Mat integralImage_2(img_2.size(), CV_32F, 1);
  GenerateIntegralImage(gray32_2, integralImage_2);
  EndTimer(&timeAccumulator, SUMMED_TABLE);

  // Compute Interest Points from summed table representation
  StartTimer(&timeAccumulator, INTEREST_POINT_DETECTION);

  std::vector<Point> interestPoints1;
  FastHessian fh_1(integralImage_1, interestPoints1, 5, 4, SAMPLING_RATE, 0.00004f);
  fh_1.getIpoints();

  std::vector<Point> interestPoints2;
  FastHessian fh_2(integralImage_2, interestPoints2, 5, 4, SAMPLING_RATE, 0.00004f);
  fh_2.getIpoints();
  EndTimer(&timeAccumulator, INTEREST_POINT_DETECTION);
  printf("Num interest points = %lu, %lu\n", interestPoints1.size(), interestPoints2.size());


  StartTimer(&timeAccumulator, DESCRIPTOR_EXTRACTION);
  // Compute Brief Descriptor based off interest points, images are greyscale, CV_32F
  FourTupleVector fourTuplevector;
  Brief briefDescriptor(fourTuplevector, PATCH_SIZE);
  briefDescriptor.ReadFile(fileName);
  vector<BriefPointDescriptor> desiciptorVector1;
  briefDescriptor.ComputeBriefDescriptor(gray32_1, interestPoints1, desiciptorVector1);
  vector<BriefPointDescriptor> desiciptorVector2;
  briefDescriptor.ComputeBriefDescriptor(gray32_2, interestPoints2, desiciptorVector2);

  EndTimer(&timeAccumulator, DESCRIPTOR_EXTRACTION);

  StartTimer(&timeAccumulator, DESCRIPTOR_MATCHING);
  // Match Interest Points
  vector<cv::Point>points_1;
  vector<cv::Point>points_2;
  FindMatches(desiciptorVector1, desiciptorVector2, points_1, points_2);

  EndTimer(&timeAccumulator, DESCRIPTOR_MATCHING);


// // OPENCV ROUTINES TO STITCH CODE, CREATE SEPARATE FUNCTION

  StartTimer(&timeAccumulator, OPENCV_ROUTINES);

  Mat homographyMat_1 = Mat::eye(3, 3, CV_32F);
  Mat homographyMat_2 = (cv::findHomography(points_2, points_1, RANSAC, 4.0));
  homographyMat_2.convertTo(homographyMat_2, CV_32F);

  double xMin = INT_MAX; 
  double yMin = INT_MAX;
  double xMax = 0;
  double yMax = 0;

  std::vector<Point2d> corners = getWarpCorners(img_1, homographyMat_1);
  for (uint32_t j = 0; j < corners.size(); j++) 
  {
    xMin = std::min(xMin, corners[j].x);
    xMax = std::max(xMax, corners[j].x);
    yMin = std::min(yMin, corners[j].y);
    yMax = std::max(yMax, corners[j].y);
  }

  corners = getWarpCorners(img_2, homographyMat_2);
  for (uint32_t j = 0; j < corners.size(); j++) 
  {
    xMin = std::min(xMin, corners[j].x);
    xMax = std::max(xMax, corners[j].x);
    yMin = std::min(yMin, corners[j].y);
    yMax = std::max(yMax, corners[j].y);
  }

  // shift the panorama if warped images are out of boundaries
  double shiftX = -xMin;
  double shiftY = -yMin;
  Mat transM = getTranslationMatrix(shiftX, shiftY);
  int width = std::round(xMax - xMin);
  int height = std::round(yMax - yMin);

  homographyMat_1 = (transM * homographyMat_1) / homographyMat_1.at<float>(1, 1);
  homographyMat_2 = (transM * homographyMat_2) / homographyMat_2.at<float>(1, 1);

  Mat panorama = Mat::zeros(height, width, img_1.type());
  Mat pano_mask = Mat::zeros(height, width, img_1.type());

  Mat img_mask1 = createMask(img_1);
  cv::warpPerspective(img_mask1, img_mask1, homographyMat_1, Size(width, height));
  panorama = stitchImages(panorama, img_1, homographyMat_1, pano_mask, img_mask1);

  pano_mask = pano_mask + img_mask1;

  Mat img_mask2 = createMask(img_2);
  cv::warpPerspective(img_mask2, img_mask2, homographyMat_2, Size(width, height));
  panorama = stitchImages(panorama, img_2, homographyMat_2, pano_mask, img_mask2);

  EndTimer(&timeAccumulator, OPENCV_ROUTINES);
  EndTimer(&timeAccumulator, TOTAL_TIME);
  PrintTimes(&timeAccumulator);

  imshow("pano", panorama);
  waitKey(0);

  return 0;
}


// pass by reference
static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage_1)
{

  integralImage_1.at<float>(0, 0) = source.at<float>(0, 0);
  for (int j = 1; j < integralImage_1.cols; ++j)
  {
    integralImage_1.at<float>(0, j) = integralImage_1.at<float>(0, j - 1) + 
    source.at<float>(0, j);
  }

  for (int i = 1; i < integralImage_1.rows; ++i)
  {
    float accumulator = 0.0f;
    for (int j = 0; j < integralImage_1.cols; ++j)
    {
      accumulator += source.at<float>(i, j);
      integralImage_1.at<float>(i, j) = accumulator + integralImage_1.at<float>(i - 1, j);
    }
  }
}


inline int fRound(float flt)
{
  return (int) floor(flt+0.5f);
}

//! Draw all the Ipoints in the provided vector
void drawIpoints(Mat img_1, vector<InterestPoint> &ipts)
{
  InterestPoint *ipt;
  int r1, c1;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    ipt = &ipts.at(i);
    r1 = fRound(ipt->position.second);
    c1 = fRound(ipt->position.first);

    circle(img_1, Point(c1,r1), 1, cvScalar(0, 255, 0),-1);
  }
}
