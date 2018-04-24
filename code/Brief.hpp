#pragma once

#include<vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include <bitset>
#include "InterestPoint.hpp"

struct BriefPointDescriptor
{
	uint64_t desc_array[4];
	int row;
	int col;

	BriefPointDescriptor() {
		desc_array[0] = 0;
		desc_array[1] = 0;
		desc_array[2] = 0;
		desc_array[3] = 0;
	}
};

struct FourTuple
{
	int x1;
	int y1;
	int x2;
	int y2;
};

typedef std::vector<FourTuple> FourTupleVector;

class Brief
{
public:
	Brief(FourTupleVector &briefPairs,int patchSize);
	~Brief();
	int ReadFile(const char* filename);
	void ComputeBriefDescriptor(const cv::Mat &img, std::vector<InterestPoint> &ipts, std::vector<BriefPointDescriptor> &desiciptorVector);

	int patchSize;
	int patchWidth;
	int numBriefPairs;
	FourTupleVector &briefPairs;
	
private:
	bool isValidPoint(const InterestPoint &ipt, float width, float height);
	void ComputeSingleBriefDescriptor(const cv::Mat &greyImage, const InterestPoint &ipt, BriefPointDescriptor &descriptor);
	
};

void FindMatches(vector<BriefPointDescriptor> &descripts1, 
  vector<BriefPointDescriptor> &descripts2, vector<cv::Point>points1,
  vector<cv::Point>points2);
