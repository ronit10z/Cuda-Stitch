#pragma once

#include<vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>

#include <bitset>
#include "InterestPoint.hpp"

struct Brief_Point_Descriptor
{
	uint64_t desc_array[4];
	int row;
	int col;

	Brief_Point_Descriptor() {
		desc_array[0] = 0;
		desc_array[1] = 0;
		desc_array[2] = 0;
		desc_array[3] = 0;
	}
};



struct Brief_Full_Descriptor
{
	std::vector<cv::Point> ipts;
	std::vector<Brief_Point_Descriptor> descriptors;
};

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
	bool isValidPoint(const InterestPoint &ipt);
	
	void computeSingleBriefDescriptor(const InterestPoint &ipt, Brief_Point_Descriptor &descript);

	void computeBriefDesciptor(std::vector<Brief_Point_Descriptor> &descripts);
	
	cv::Mat &greyImage;
	std::vector<InterestPoint> &ipts;
	int width;
	int height;
	int patchSize;
	int patchWidth;
	
	int numBriefPairs;
	FourTupleVector &briefPairs;
};
