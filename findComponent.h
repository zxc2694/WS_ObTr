#ifndef FINDCOMPONENT_H
#define FINDCOMPONENT_H

#include "MeanShiftTracker.h"
#include "ObjectTrackerFactory.h"

void find_connected_components(IplImage *mask, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers);
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location);
int Overlap(Rect a, Rect b, float ration);
void MorphologyProcess(IplImage* &fgmaskIpl);
void BubbleSort(int* array, int size);

#endif 