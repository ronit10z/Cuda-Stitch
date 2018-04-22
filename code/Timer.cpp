#include <iostream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include "Timer.hpp"


TimeAccumulator timeAccumulator;
const char* ACTIVITY_NAMES[NUM_ACTIVITIES] {"TOTAL_TIME", "IMAGE_CONVERSION", "SUMMED_TABLE", "RESPONSES", "NMS"};

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
    printf("%lf ms         %lf      %s\n", time, currentPercent, ACTIVITY_NAMES[i]);
  }

  printf("%lf per-cent accounted for.\n", totalPercentageAccounted);
  printf("\nTOTAL TIME TAKEN %lf \n", totalTime);
}