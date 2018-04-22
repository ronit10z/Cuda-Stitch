#include <fstream>

#include "Brief.hpp"

using namespace std;
using namespace cv;

Brief::Brief(cv::Mat &img, std::vector<InterestPoint> &ipts, FourTupleVector &briefPairs,int patchSize):
greyImage(img), ipts(ipts), briefPairs(briefPairs)
{
	this->width = img.cols;
	this->height = img.rows;
	this->patchSize = patchSize;
	this->patchWidth = patchSize / 2;
	this->numBriefPairs = -1;
}

Brief::~Brief()
{

}

void Brief::ReadFile(const char* filename)
{
	ifstream inFile;
	infile.open(filename);
	
	if (!inFile) {
    cout << "Unable to open file";
    exit(1); // terminate with error
  }
}