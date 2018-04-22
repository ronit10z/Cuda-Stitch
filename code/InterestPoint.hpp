#pragma once

#include <utility>

using namespace std;
#define NUM_DIMS 64

class InterestPoint
{
public:
	InterestPoint();
	~InterestPoint();

	pair<float, float> position;
	float scale;
	float angle;
	int laplaceValue;
};