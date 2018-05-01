#include <iostream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include "Timer.hpp"


TimeAccumulator timeAccumulator;
const char* ACTIVITY_NAMES[NUM_ACTIVITIES] {"TOTAL_TIME", "IMAGE_IO", "SUMMED_TABLE", "DET_CAL", "NMS", "INTEREST_POINT_DETECTION", 
  "DESCRIPTOR_EXTRACTION", "DESCRIPTOR_MATCHING", "OPENCV_ROUTINES"};

static inline TIME currentSeconds()
{
  return std::chrono::high_resolution_clock::now();
}

void SetupTimer(TimeAccumulator* timeAccumulator)
{
  for (int i = 0; i < NUM_ACTIVITIES; ++i)
  {
    timeAccumulator->timeTaken[i] = 0.0f;
  }
}

void StartTimer(TimeAccumulator* timeAccumulator, Activity activityNumber)
{
  if (activityNumber >= NUM_ACTIVITIES) printf("Exceeded index\n");
  timeAccumulator->startTimes[activityNumber] = currentSeconds();
}

void EndTimer(TimeAccumulator* timeAccumulator, Activity activityNumber)
{
  if (activityNumber >= NUM_ACTIVITIES) printf("Exceeded index\n");
  timeAccumulator->timeTaken[activityNumber] += (std::chrono::duration<double, std::milli>(currentSeconds() - timeAccumulator->startTimes[activityNumber]).count());
}


void PrintTimes(TimeAccumulator* timeAccumulator)
{
  int i;
  double totalTime = timeAccumulator->timeTaken[TOTAL_TIME];
  double totalPercentageAccounted = 0;

  printf("****************BREAK DOWN OF TIME****************\n");
  for (i = 1; i < NUM_ACTIVITIES; ++i)
  {
    double time = timeAccumulator->timeTaken[i];
    double currentPercent = (time / totalTime) * 100;
    totalPercentageAccounted += currentPercent;
    printf("%.4f ms         %.4f      %s\n", time, currentPercent, ACTIVITY_NAMES[i]);
  }

  printf("%.4f per-cent accounted for.\n", totalPercentageAccounted);
  printf("\nTOTAL TIME TAKEN %.4f \n", totalTime);
}