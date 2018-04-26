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

//-------------------------------------------------------

static inline void GenerateIntegralImage(const Mat &source, Mat &integralImage_1);
void drawIpoints(Mat img_1, vector<InterestPoint> &ipts);
static inline void GetPanSize(double &xMin, double &xMax, double &yMin, double &yMax, Mat &img, Mat &homographyMat);
Mat getTranslationMatrix(float x, float y);
void stitch(int height, int width, Mat &img_1, Mat &img_2, 
  Mat &homographyMat_1, Mat &homographyMat_2);


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

int main(int argc, char const *argv[])
{
  // TODO: change filename to command line arguement, have assert or 
  // exit if argc is not enough.
  const char* fileName = "../../../../ParaPano/data/testPattern.txt";
  Mat img1 = imread("../../images/bryce_left_02.png", IMREAD_GRAYSCALE);
  Mat img2 = imread("../../images/bryce_right_02.png", IMREAD_GRAYSCALE);
  // img_1.convertTo(img_1, CV_32F);
  // img_2.convertTo(img_2, CV_32F);
  Mat img_1, img_2;
  img1.convertTo(img_1, CV_32F);
  img2.convertTo(img_2, CV_32F);




  int minHessian = 400;
  Ptr<SURF> detector = SURF::create( minHessian );
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  detector->detect( img1, keypoints_1 );
  detector->detect( img2, keypoints_2 );
  std::vector<Point> interestPoints1;
  std::vector<Point> interestPoints2;
  for (uint32_t i = 0; i < keypoints_1.size(); ++i)
  {
    interestPoints1.push_back(Point(keypoints_1[i].pt.x, keypoints_1[i].pt.y));
  }
  for (uint32_t i = 0; i < keypoints_2.size(); ++i)
  {
    interestPoints2.push_back(Point(keypoints_2[i].pt.x, keypoints_2[i].pt.y));
  }

  // Compute Brief Descriptor based off interest points
  FourTupleVector fourTuplevector;
  Brief briefDescriptor(fourTuplevector, PATCH_SIZE);
  briefDescriptor.ReadFile(fileName);
  // end of setup for brief desciptor, need to feed in new image only, if you want to do
  // it for another image, the parsing is expensive but only happens once per run.

  vector<BriefPointDescriptor> desiciptorVector1;
  briefDescriptor.ComputeBriefDescriptor(img_1, interestPoints1, desiciptorVector1);
  vector<BriefPointDescriptor> desiciptorVector2;
  briefDescriptor.ComputeBriefDescriptor(img_2, interestPoints2, desiciptorVector2);
  // get both desciptors
  printf("Number of Interest Points Detected in im1 = %u\n", (uint32_t)interestPoints1.size());
  printf("descriptVector1 length = %lu\n", desiciptorVector1.size());
  printf("descriptVector2 length = %lu\n", desiciptorVector2.size());

  for (uint32_t i = 0; i < desiciptorVector1.size(); ++i)
  {
    printf("%lu %lu %lu %lu\n", desiciptorVector1[i].desc_array[0], desiciptorVector1[i].desc_array[1], 
      desiciptorVector1[i].desc_array[2], desiciptorVector1[i].desc_array[3]);
  }
  exit(1);

  vector<cv::Point>points_1;
  vector<cv::Point>points_2;
  FindMatches(desiciptorVector1, desiciptorVector2, points_1, points_2);

  // for (uint32_t i = 0; i < points_2.size(); ++i)
  // {
  //   printf("%d %d\n", points_2[i].x, points_2[i].y);
  // }


  // exit(1);

  Mat homographyMat_1 = (Mat::eye(3, 3, CV_32F));
  Mat homographyMat_2 = cv::findHomography(points_2, points_1, cv::RANSAC, 4.0);
  homographyMat_2.convertTo(homographyMat_2, CV_32F);
  homographyMat_2 = homographyMat_1 * homographyMat_2;
  
  double xMin, yMin = INT_MAX;
  double xMax, yMax = 0;
  GetPanSize(xMin, xMax, yMin, yMax, img_1, homographyMat_1);
  GetPanSize(xMin, xMax, yMin, yMax, img_2, homographyMat_2);

  double shiftX = -xMin;
  double shiftY = -yMin;
  Mat transM = getTranslationMatrix(shiftX, shiftY);

  // // // initialize empty panorama

  homographyMat_1 = (homographyMat_1*transM) / homographyMat_1.at<float>(2, 2);
  homographyMat_2 = (homographyMat_2*transM) / homographyMat_2.at<float>(2, 2);

  int width = std::round(xMax - xMin);
  int height = std::round(yMax - yMin);
  Mat imcolor1 = imread("../../images/bryce_left_02.png", IMREAD_COLOR);
  Mat imcolor2 = imread("../../images/bryce_right_02.png", IMREAD_COLOR);

  stitch(height, width, imcolor1, imcolor2, homographyMat_1, homographyMat_2);

  return 0;
}

Mat createMask(Mat& im) 
{
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

Mat stitchImages(Mat& pano, Mat& image, Mat& H, Mat& pano_mask, Mat& img_mask) 
{
  int width = pano.cols;
  int height = pano.rows;

  Mat image_warped;
  image_warped.create(image.size(), image.type());

  warpPerspective(image, image_warped, H, Size(width, height));

  Mat bim1, bim2;
  multiply(pano, pano_mask, bim1);
  img_mask.convertTo(img_mask, image_warped.type());
  multiply(image_warped, img_mask, bim2);

  Mat stitch_img;
  divide(bim1 + bim2, pano_mask + img_mask, stitch_img);
  return stitch_img;
}

void stitch(int height, int width, Mat &img_1, Mat &img_2, 
  Mat &homographyMat_1, Mat &homographyMat_2)
{
  Mat panorama = Mat::zeros(height, width, img_1.type());
  Mat pano_mask = Mat::zeros(height, width, img_1.type());

  Mat imgMask_1 = createMask(img_1);
  cv::warpPerspective(imgMask_1, imgMask_1, homographyMat_1, Size(width, height));
  // BUG ON NEXT LINE
  panorama = stitchImages(panorama, img_1, homographyMat_1, pano_mask, imgMask_1);
  pano_mask += imgMask_1;

  Mat imgMask_2 = createMask(img_2);
  cv::warpPerspective(imgMask_2, imgMask_2, homographyMat_2, Size(width, height));
  panorama = stitchImages(panorama, img_2, homographyMat_2, pano_mask, imgMask_2);

  imshow("Panorama", panorama);

  waitKey(0);
}


cv::Mat getTranslationMatrix(float x, float y) 
{
    Mat M = Mat::eye(3, 3, CV_32F);
    M.at<float>(0,2) = x;
    M.at<float>(1,2) = y;
    return M;
}

std::vector<Point2d> getWarpCorners(Mat& im, Mat& H) 
{
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

static inline void GetPanSize(double &xMin, double &xMax, double &yMin, double &yMax, Mat &img, Mat &homographyMat)
{
  std::vector<Point2d> corners = getWarpCorners(img, homographyMat);
  for (uint32_t i = 0; i < corners.size(); i++) 
  {
    xMin = std::min(xMin, corners[i].x);
    xMax = std::max(xMax, corners[i].x);
    yMin = std::min(yMin, corners[i].y);
    yMax = std::max(yMax, corners[i].y);
  }
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
