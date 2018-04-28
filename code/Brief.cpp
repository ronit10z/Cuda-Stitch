#include <fstream>
#include <climits>
#include <cmath>

#include "Brief.hpp"

using namespace std;
using namespace cv;

#define THRESHOLD 0.8

void PrintBriefDescriptor(BriefPointDescriptor &descript) {
  printf("%lu %lu %lu %lu\n", descript.desc_array[0], descript.desc_array[1], 
  descript.desc_array[2], descript.desc_array[3]);
}
// sets the bit corresponding to the index to val.
static inline void BriefSet(int index, bool val, BriefPointDescriptor &descriptor, int k) {
  if (index < 0 || index > 255) 
  {
    printf("Index out of bounds for 256 bit descriptor");
    return;
  }
  if (val == 1) 
  {
    //calculate the array element we need to go to
    int idx = index / 64; //idx is the element we need to be inside
    int offset = index % 64; //offset is the position inside of the element

    long mask = 1L << offset;
    descriptor.desc_array[idx] |= mask;
  }
  // if (k == 119) PrintBriefDescriptor(descriptor);

}

//spit out the descriptor in bit form on cout for debugging if needed

// Just save all the arguements.
Brief::Brief(FourTupleVector &briefPairs,int patchSize):
briefPairs(briefPairs)
{
	this->patchSize = patchSize;
	this->patchWidth = patchSize / 2;
	this->numBriefPairs = -1;
}

Brief::~Brief()
{

}

// Takes in file, reads the number of brief pairs at the top of the file an then
// reserves the space in the vector and parses in the contents of the file
int Brief::ReadFile(const char* filename)
{
	ifstream infile;
	infile.open(filename);
	
	if (!infile) {
    cout << "Unable to open file";
    exit(1); // terminate with error
  }
  	//get the number of pairs that we want to use to form the descriptor
  	infile >> (this->numBriefPairs);
    this->briefPairs.resize(this->numBriefPairs);
  	
  	int x1, y1, x2, y2;
  	for (int i = 0; i < this->numBriefPairs; i++) {
  		infile >> x1 >> y1 >> x2 >> y2;

      // cout << x1 << " " << y1 << " "  << x2 << " " << y2;
      // exit(1);
  		
      this->briefPairs[i].x1 = x1;
      this->briefPairs[i].y1 = y1;
      this->briefPairs[i].x2 = x2;
      this->briefPairs[i].y2 = y2;

  	}

  return this->numBriefPairs;
}

inline bool isInBound(int r, int c, int h, int w) 
{
  return r >= 0 && r < h && c >= 0 && c < w;
}

// bool hasValidPatch(int h, int w, int row, int col) 
// {
//   int r = PATCH_SIZE / 2;
//   return isInBound(row - r, col - r, h, w) && isInBound(row + r, col + r,h,w);
// }

//Determine if the point is within bounds
bool Brief::isValidPoint(const Point &ipt, int width, int height)
{
  int col = (ipt.x);
  int row = (ipt.y);
  // int dist = (this->patchWidth);
  int dist = 4;

  return isInBound(row - dist, col - dist, height, width) && isInBound(row + dist, col + dist,height,width); 
}



// loops over all the points and calculates the descripter if the point is within bounds.
void Brief::ComputeBriefDescriptor(const cv::Mat &img, std::vector<Point> &ipts, std::vector<BriefPointDescriptor> &desiciptorVector)
{ 
  desiciptorVector.resize(ipts.size());
  int j = 0;
  int width = (img.cols);
  int height = (img.rows);
  Mat blurredIm;
  Mat temp = img / 255.0;
  cv::GaussianBlur(temp, blurredIm, Size(5, 5), 1/sqrt(2), 1/sqrt(2));

  for (uint32_t k = 0; k < ipts.size(); k++)
  {
    Point &ipoint = ipts[k];
    int col = ipoint.x;
    int row = ipoint.y;

    // float patch[9][9];
    if (isValidPoint(ipoint, width, height))
    {
      // for (int i = 0; i < patchSize; ++i)
      // {
      //   for (int j = 0; j < patchSize; ++j)
      //   {
      //     patch[i][j] = blurredIm.at<float>(row + i - patchSize/2, col + j - patchSize/2);
      //   }
      // }
      // TODO: WHY IS THE RUM ALWAYS GONE, why is patch faster?

      desiciptorVector[j].col = col;
      desiciptorVector[j].row = row;

      for (int idx = 0; idx < this->numBriefPairs; idx++)
      {
        int x1 = briefPairs[idx].x1 + col;
        int y1 = briefPairs[idx].y1 + row;
        int x2 = briefPairs[idx].x2 + col;
        int y2 = briefPairs[idx].y2 + row;

        float pixel1 = blurredIm.at<float>(y1, x1);
        float pixel2 = blurredIm.at<float>(y2, x2);

        BriefSet(idx, pixel1 < pixel2, desiciptorVector[j], k);
      }
      j++;
    }
  }
  desiciptorVector.resize(j);
}

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

// returns the hamming distance between two descripters
static inline int FindDistance(BriefPointDescriptor &descriptor1, BriefPointDescriptor &descriptor2)
{
  uint64_t xor1 = descriptor1.desc_array[0] ^ descriptor2.desc_array[0];
  uint64_t xor2 = descriptor1.desc_array[1] ^ descriptor2.desc_array[1];   
  uint64_t xor3 = descriptor1.desc_array[2] ^ descriptor2.desc_array[2];   
  uint64_t xor4 = descriptor1.desc_array[3] ^ descriptor2.desc_array[3];

  return CountOnes(xor1) + CountOnes(xor2) + CountOnes(xor3) + CountOnes(xor4);
}

// pass in the descipts for the two things you want to match, as well as a reference to two empty cv::Point vectors
void FindMatches(vector<BriefPointDescriptor> &descripts1, 
  vector<BriefPointDescriptor> &descripts2, vector<cv::Point> &points1,
  vector<cv::Point> &points2)
{
  for (uint32_t i = 0; i < descripts1.size(); ++i)
  {
    BriefPointDescriptor &descriptor1 = descripts1[i];
    int bestDistance = INT_MAX;
    int secondBest = INT_MAX;
    int bestj = 0;

    for (uint32_t j = 0; j < descripts2.size(); ++j)
    {
      BriefPointDescriptor &descriptor2 = descripts2[j];
      int distance = FindDistance(descriptor1, descriptor2);

      // printf("%d %d\n", j, distance);
      if (distance < bestDistance)
      {
        bestj = j;
        secondBest = bestDistance;
        bestDistance = distance;
      }
      else if(distance < secondBest) secondBest = distance;
    }
    float ratio;
    if (secondBest == 0) ratio = bestDistance;
    else ratio = static_cast<float>(bestDistance) / static_cast<float>(secondBest);  
    
    // printf("best Distance = %d, secondBest = %d, ratio = %f\n", bestDistance, secondBest, ratio);
    if (ratio < THRESHOLD)
    {
      // printf("best j = %d\n", bestj);
      points1.push_back(Point(descripts1[i].col, descripts1[i].row));
      points2.push_back(Point(descripts2[bestj].col, descripts2[bestj].row));
    }
    // exit(1);
  }
}