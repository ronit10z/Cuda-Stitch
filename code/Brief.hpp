#pragma once
#include<vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include "InterestPoint.hpp"

typedef struct
{
	int x1;
	int y1;
	int x2;
	int y2;
}FourTuple;

typedef std::vector<FourTuple> FourTupleVector;

std::vector<cv::Point> GetBriefPatch(int row, int col, int patchSize);

class Brief
{
public:
	Brief(cv::Mat &img, std::vector<InterestPoint> &ipts, FourTupleVector &briefPairs,int patchSize);
	~Brief();
	void ReadFile(const char* filename);
	bool isValidPoint(InterestPoint ipt);
	
	cv::Mat &greyImage;
	std::vector<InterestPoint> &ipts;
	int width;
	int height;
	int patchSize;
	int patchWidth;
	
	int numBriefPairs;
	FourTupleVector &briefPairs;
};
