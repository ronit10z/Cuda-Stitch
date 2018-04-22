#pragma once

#include <opencv/cv.h>
#include "InterestPoint.hpp"
#include "ResponseLayer.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include <vector>

static const int OCTAVES = 5;
static const int INTERVALS = 4;
static const float THRES = 0.0004f;
static const int INIT_SAMPLE = 2;


class FastHessian {
  
  public:
   
    FastHessian(Mat img, 
                std::vector<InterestPoint> &ipts, 
                const int octaves, 
                const int intervals, 
                const int init_sample, 
                const float thres);

    ~FastHessian();

    void getIpoints();
    
  private:

    //---------------- Private Functions -----------------//

    //! Build map of DoH responses
    void buildResponseMap();

    //! Calculate DoH responses for supplied layer
    void buildResponseLayer(ResponseLayer *r);

    //! 3x3x3 Extrema test
    int isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);    
    
    //! Interpolation functions - adapted from Lowe's SIFT implementation
    void interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
    void interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,
                          double* xi, double* xr, double* xc );
    CvMat* deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
    CvMat* hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);

    //---------------- Private Variables -----------------//

    Mat img;
    int i_width, i_height;

    std::vector<InterestPoint> &ipts;

    std::vector<ResponseLayer *> responseMap;

    int octaves;

    int intervals;

    int init_sample;

    float thresh;
};