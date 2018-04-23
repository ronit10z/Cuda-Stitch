#include <fstream>

#include "Brief.hpp"

using namespace std;
using namespace cv;

//set a bit in the brief desciptor, where index is a position in 0..255 inclusive
//and val is the value to set the bit to
void brief_set(int index, bool val, Brief_Point_Descriptor &descript) {
  if (index < 0 || index > 255) {
    printf("Index out of bounds for 256 bit descriptor");
    return;
  }
  if (val == 1) {
    //calculate the array element we need to go to
    int idx = index / 64; //idx is the element we need to be inside
    int offset = index % 64; //offset is the position inside of the element

    int mask = 1L << offset;
    descript.desc_array[idx] |= mask;
  }
}

//spit out the descriptor in bit form on cout for debugging if needed
void print_brief_descriptor(Brief_Point_Descriptor &descript) {
  std::cout << std::bitset<64>(descript.desc_array[3]) << std::bitset<64>(descript.desc_array[2])
    << std::bitset<64>(descript.desc_array[1]) << std::bitset<64>(descript.desc_array[0]);
}

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

//Read from the given file, which contains pairs of points to use while
//calculating the brief descriptor
void Brief::ReadFile(const char* filename)
{
	ifstream infile;
	infile.open(filename);
	
	if (!infile) {
    cout << "Unable to open file";
    exit(1); // terminate with error
  }
  	//get the number of pairs that we want to use to form the descriptor
  	infile >> (this->numBriefPairs);
  	
  	int x1, y1, x2, y2;
  	for (int i = 0; i < this->numBriefPairs; i++) {
  		infile >> x1 >> y1 >> x2 >> y2;
  		
      this->briefPairs[i].x1 = x1;
      this->briefPairs[i].y1 = y1;
      this->briefPairs[i].x2 = x2;
      this->briefPairs[i].y2 = y2;

  	}

}

//Determine if the patch of the point (given patch size) is in the image scope
bool Brief::isValidPoint(const InterestPoint &ipt) 
{
  float col = ipt.position.first;
  float row = ipt.position.second;
  float dist = static_cast<float>(this->patchWidth);
  float width = static_cast<float>(this->width);
  float height = static_cast<float>(this->height);

  return (row - dist >= 0 && col - dist >= 0 && row + dist <= height && col + dist <= width);
}

//Given an image and an interestpoint and the list of points to compare, compute one
//BRIEF descriptor for that interestpoint

//technically don't need to move the entire img into here
//could store the values in the patch around the point somewhere and just move that
//around
void Brief::computeSingleBriefDescriptor(const InterestPoint &ipt, Brief_Point_Descriptor &descript)
{
  int r = ipt.position.first;
  int c = ipt.position.second;

  descript.row = r;
  descript.col = c;

  for (int idx = 0; idx < this->numBriefPairs; idx++)
  {
    //x1..y2 are all between -4 and 4, inclusive
    //representing relative offsets from r,c
    int x1 = briefPairs[idx].x1;
    int y1 = briefPairs[idx].y1;
    int x2 = briefPairs[idx].x2;
    int y2 = briefPairs[idx].y2;

    float pixel1 = greyImage.at<float>(r + x1, c + y1);
    float pixel2 = greyImage.at<float>(r + x2, c + y2);

    //set bit in brief descriptor according to comparison of pixel1 and pixel2
    brief_set(idx, pixel1 < pixel2, descript);
  }
}


//Given an image, a vector of interest points, and the lists of points to compare for BRIEF
//return an object containing the BRIEF descriptor for each interest point
void Brief::computeBriefDesciptor(std::vector<Brief_Point_Descriptor> &descripts)
{
  //need to loop over interestpoints, and run computeSingleBriefDescriptor, 
  //and push results on some vectors

  std::vector<InterestPoint>::iterator it;
  
  for (uint32_t i = 0; i < ipts.size(); i++)
  {
    InterestPoint &ipoint = this->ipts[i];
    Brief_Point_Descriptor &descript = descripts[i];

    if (isValidPoint(ipoint))
    {
      computeSingleBriefDescriptor(ipoint, descript);
    }
  }
}




