#ifndef OBJECTFORM_H
#define OBJECTFORM_H

#include "opencv2/core/core.hpp"

using namespace cv;

const short MaxHistBins = 4096;

typedef struct
{
	int No;
	short	type;				// 1: pedestrian, 2: vehicle, 3: unknown
	short	status;				// 1: detected, 2: tracked, 3: miss to detect, 4: loss to track
	Point2d cen_pos;
	CvRect	boundingBox;		// in pixels
	float	hist[MaxHistBins];	// disparity(32_bins) + intensity(32_bins) : for tracking
	double	similar_val;		// value of similarity function
	float	minDisparity;
	float	maxDisparity;
	float	medianDisparity;
	float	meanDisparity;
	float	stdDisparity;
	Point3f xyz0;				// 3d position of previous time instance in world coordinate, minimum distance from camera
	Point3f xyz;				// 3d position of currnet time in world coordinate, minimum distance from camera
	Size	objSize;
	vector<float> descriptor;
	Point point[100];
	Scalar color;
	int PtNumber;
	int PtCount;
	char Run;
} Object2D;

typedef struct
{
	short	type;				// 1: pedestrian, 2: vehicle, 3: unknown
	short	status;				// 1: detected, 2: tracked, 3: miss to detect, 4: loss to track
	Rect	boundingBox;		// in pixels
	Point3f xyz0;				// 3d position of previous time instance in world coordinate, minimum distance from camera
	Point3f xyz;				// 3d position of currnet time in world coordinate, minimum distance from camera
	float	hist[MaxHistBins];	// disparity(32_bins) + intensity(32_bins) : for tracking
	vector<float> descriptor;
} Object3D;

#endif