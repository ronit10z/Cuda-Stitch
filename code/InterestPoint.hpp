#pragma once

#include <utility>

using namespace std;

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