#ifndef CYCLETIMER_H

#define TIMER_DEBUG
#define NUM_ACTIVITIES 5

#include <chrono>

typedef enum {TOTAL_TIME, IMAGE_CONVERSION, SUMMED_TABLE, RESPONSES, NMS} Activity;
typedef std::chrono::_V2::system_clock::time_point TIME;


typedef struct
{
	TIME startTimes[NUM_ACTIVITIES];
  double timeTaken[NUM_ACTIVITIES];
}TimeAccumulator;


void SetupTimer(TimeAccumulator* timeAccumulator);
void StartTimer(TimeAccumulator* timeAccumulator, Activity activityNumber);
void EndTimer(TimeAccumulator* timeAccumulator, Activity activityNumber);
void PrintTimes(TimeAccumulator* timeAccumulator);

#define CYCLETIMER_H
#endif