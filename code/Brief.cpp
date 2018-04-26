#include <fstream>
#include <climits>
#include <cmath>

#include "Brief.hpp"

using namespace std;
using namespace cv;

#define THRESHOLD 0.8

// sets the bit corresponding to the index to val.
static inline void BriefSet(int index, bool val, BriefPointDescriptor &descriptor) {
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

    int mask = 1L << offset;
    descriptor.desc_array[idx] |= mask;
  }
}

//spit out the descriptor in bit form on cout for debugging if needed
void PrintBriefDescriptor(BriefPointDescriptor &descript) {
  printf("%lu %lu %lu %lu\n", descript.desc_array[0], descript.desc_array[1], 
  descript.desc_array[2], descript.desc_array[3]);
}

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
  int dist = (this->patchWidth);

  return isInBound(row - dist, col - dist, height, width) && isInBound(row + dist, col + dist,height,width); 
}



// loops over all the points and calculates the descripter if the point is within bounds.
void Brief::ComputeBriefDescriptor(const cv::Mat &img, std::vector<Point> &ipts, std::vector<BriefPointDescriptor> &desiciptorVector)
{ 
  desiciptorVector.resize(ipts.size());
  int j = 0;
  int width = (img.cols);
  int height = (img .rows);
  Mat blurredIm;
  Mat temp = img / 255.0;
  cv::GaussianBlur(temp, blurredIm, Size(5, 5), 1/sqrt(2), 1/sqrt(2));

  for (uint32_t k = 0; k < ipts.size(); k++)
  {
    Point &ipoint = ipts[k];
    int col = ipoint.x;
    int row = ipoint.y;

    float patch[9][9];
    if (isValidPoint(ipoint, width, height))
    {
      // printf("initial =   %d %d\n", row, col);
      for (int i = 0; i < patchSize; ++i)
      {
        for (int j = 0; j < patchSize; ++j)
        {
          // printf("%d %d (%d %d) =  %f\n", i, j, row + i - patchSize/2, col + j - patchSize/2, blurredIm.at<float>(row + i - patchSize/2, col + j - patchSize/2));
          patch[i][j] = blurredIm.at<float>(row + i - patchSize/2, col + j - patchSize/2);
        }
      }

      desiciptorVector[j].col = col;
      desiciptorVector[j].row = row;

      for (int idx = 0; idx < this->numBriefPairs; idx++)
      {
        int x1 = briefPairs[idx].x1;
        int y1 = briefPairs[idx].y1;
        int x2 = briefPairs[idx].x2;
        int y2 = briefPairs[idx].y2;

        float pixel1 = patch[y1][x1];
        float pixel2 = patch[y2][x2];

        // printf("(%d, %d) = %f | (%d, %d) = %f\n", x1, y1, pixel1, x2, y2, pixel2);
        printf("%d %d\n", idx, pixel1 < pixel2);
        BriefSet(idx, pixel1 < pixel2, desiciptorVector[j]);
      }
      printf("row = %d  col = %d ", row, col);
      PrintBriefDescriptor(desiciptorVector[j]);
      j++;
    }
    exit(1);
  }
  desiciptorVector.resize(j);
}


// Calulates the descipter for a given point
void Brief::ComputeSingleBriefDescriptor(const cv::Mat &greyImage, const Point &ipt, BriefPointDescriptor &descriptor)
{
  // position is (x, y)
  int c = ipt.x;
  int r = ipt.y;

  descriptor.col = ipt.x;
  descriptor.row = ipt.y;

  for (int idx = 0; idx < this->numBriefPairs; idx++)
  {
    int x1 = briefPairs[idx].x1;
    int y1 = briefPairs[idx].y1;
    int x2 = briefPairs[idx].x2;
    int y2 = briefPairs[idx].y2;

    float pixel1 = greyImage.at<float>(r + x1, c + y1);
    float pixel2 = greyImage.at<float>(r + x2, c + y2);

    BriefSet(idx, pixel1 < pixel2, descriptor);
  }
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
  int xor1 = descriptor1.desc_array[0] ^ descriptor2.desc_array[0];
  int xor2 = descriptor1.desc_array[1] ^ descriptor2.desc_array[1];   
  int xor3 = descriptor1.desc_array[2] ^ descriptor2.desc_array[2];   
  int xor4 = descriptor1.desc_array[3] ^ descriptor2.desc_array[3];

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

      if (distance < bestDistance)
      {
        bestj = j;
        secondBest = bestDistance;
        bestDistance = distance;
      }
      else if(distance < secondBest) secondBest = distance;
    }
    float ratio = static_cast<float>(bestDistance) / static_cast<float>(secondBest);
    if (secondBest == 0) ratio = 0;
    // printf("best Distance = %d, secondBest = %d, ratio = %f\n", bestDistance, secondBest, ratio);
    if (ratio < THRESHOLD)
    {
      // printf("Match !\n");
      points1.push_back(Point(descripts1[i].col, descripts1[i].row));
      points2.push_back(Point(descripts2[bestj].col, descripts2[bestj].row));
    }
  }
}