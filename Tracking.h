#ifndef __OBJECTTRACKING_H__
#define __OBJECTTRACKING_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <iomanip> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

#if (defined(_WIN32) | defined(_WIN64))
#ifdef EXPORT_CLASS_OBJECTTRACKING
#define CLASS_OBJECTTRACKING __declspec(dllexport)
#else
#define CLASS_OBJECTTRACKING __declspec(dllimport)
#endif
#else
#define CLASS_OBJECTTRACKING
#endif

/* Tracking parameters */
#define stopTrackingObjWithTooSmallWidth_Scale 120 // Delete too small tracking obj when its width becomes < (imgWidth + imgHeight) / stopTrackingObjWithTooSmallWidth_Scale
#define stopTrackingObjWithTooSmallHeight_Scale 28 // Delete too small tracking obj when its height becomes < (imgWidth + imgHeight) / stopTrackingObjWithTooSmallHeight_Scale
#define MAX_DIS_BET_PARTS_OF_ONE_OBJ  38           // Allowed max distance between the box and other tracking boxes
#define MAX_OBJ_LIST_SIZE            100           // Allowed max number of objects 
#define minObjWidth_Ini_Scale         60           // if obj bbs found by bbsFinder has width < (imgWidth + imgHeight) / minObjWidth_Ini_Scale, then addTrackedList don't add it into object_list to track it
#define minObjHeight_Ini_Scale        14           // if obj bbs found by bbsFinder has height < (imgWidth + imgHeight) / minObjHeight_Ini, then addTrackedList don't add it into object_list to track it
#define ACCEPTABLE_SIMILARITY        0.5           // if similarity < "ACCEPTABLE_SIMILARITY", stop tracking this obj, i.e. delete this obj from object_list 
#define Pixel32S(img,x,y) ((int*)img.data)[(y)*img.cols + (x)] // Get two original tracking boxes'distance

/* Display */
#define plotLineLength     99  // Set tracking line length, (allowed range: 0~99)
#define imgCompressionScale 2  // Enlarge the size of bbs X times
#define DELE_RECT_FRAMENO   4  // Allowed frames for boxes of loiter (suggest range: 5~15)
#define moveRate            2  // It's used for modifying the moving rate of predicted objects in occSolve 3. (Range:2~10)
#define keepTrajectory      0  // 0: not keep, 1: keep. (by color hist)
#define setPointY           4  // Proportional position. 0: Top of the head, 10: Soles of the feet (Range:0~10)
#define demoMode            1  // Without accumulating number (0:debug mode, 1:demo mode) 
extern int occSolve;           // 0: not use, 1: use color hist, 2: directly exchange, 3: directly exchange with prediction

/* Math */
#define PI 3.141592653589793238463 

const short MaxHistBins = 4096 + 1;

typedef struct
{
	int No;						// numbers of track boxes 
	short	type;				// 1:vehicle , 2: pedestrian, 3: unknown
	short	status;				// 1: detected, 2: tracked, 3: miss to detect, 4: loss to track
	Rect	boundingBox;		// in pixels
	int     initialBbsWidth;    // in pixels
	int     initialBbsHeight;   // in pixels
	double	hist[MaxHistBins];	// disparity(32_bins) + intensity(32_bins) : for tracking
	double	similar_val;		// value of similarity function
	double	minDisparity;
	double	maxDisparity;
	double	medianDisparity;
	double	meanDisparity;
	double	stdDisparity;
	Point3f xyz0;				// 3d position of previous time instance in world coordinate, minimum distance from camera
	Point3f xyz;				// 3d position of currnet time in world coordinate, minimum distance from camera
	Size objSize;
	vector<double> descriptor;
	Point point[100];           // trajectory points 
	Scalar color;               // bbs color 
	Mat kernelDownScale;        // kernel for the down-scaled bbs
	Mat kernel;                 // kernel for the bbs
	Mat kernelUpScale;          // kernel for the up-scaled bbs
	float objScale;
	int PtNumber;
	int cPtNumber;
	int PtCount;
	int findBbs[DELE_RECT_FRAMENO]; //It decides whether rectangles is motionless or not.
	bool bIsDrawing;			    //It decides whether trajectory is plotted or not.
	bool bIsUpdateTrack;
	int waitFrame;
	int initFrame;
	int pre_data_X;
	int pre_data_Y;
	double histV2[MaxHistBins];
	Mat kernelV2;
	char moveDirect;           //U:up, D:down, L:left, R:right
	int movement;              // 1:up, 2:up+right, 3:right, 4:right+down, 5:down, 6;down+left, 7:left, 8:left+up 
	bool startOcc;
} ObjTrackInfo;

typedef struct
{
	Rect boundingBox;
	bool bIsTrigger;
} InputObjInfo;

typedef struct
{
	float value;
	int objNum;
} OverlapCompare;

class CLASS_OBJECTTRACKING CObjectTracking
{
public:
	CObjectTracking();
	~CObjectTracking();

	Mat DistMat;
	IplImage fgmaskIpl;
	bool plotTraj, plotTrackROI;
	// don't tracking too small obj 
	int minObjWidth_Ini;
	int minObjHeight_Ini;

	// del too small obj 
	int minObjWidth;
	int minObjHeight;

	int DistBetObj(Rect a, Rect b);
	void ObjectTrackingProcessing(Mat &img_input, Mat &img_output, Mat &fgmask_input, CvRect *bbs, int ObjNum, InputObjInfo *trigROI, vector<ObjTrackInfo> &object_list);
	void addTrackedList(const Mat &img, vector<ObjTrackInfo> &object_list, Rect bbs, short type);
	void updateObjBbs(const Mat &img, vector<ObjTrackInfo> &object_list, Rect bbs, int idx);
	void drawTrackBox(Mat &img, vector<ObjTrackInfo> &object_list);
	void drawTrackTrajectory(Mat &TrackingLine, vector<ObjTrackInfo> &object_list, size_t &obj_list_iter);
	int track(Mat &img, vector<ObjTrackInfo> &object_list);
	void modifyTrackBox(Mat img_input, vector<ObjTrackInfo> &object_list, CvRect *bbs, int ObjNum);
	void occlusionNewObj(Mat img_input, vector<ObjTrackInfo> &object_list, CvRect *bbs, int ObjNum);

	void revertBbsSize(Mat &img_input, CvRect *bbs, int &ObjNum);
	void ObjNumArr(int *objNumArray, int *objNumArray_BS);
	void mergeBbsAndGetNewObjBbs(Mat img_input, vector<ObjTrackInfo> &object_list, CvRect *bbs, int ObjNum);
	void findTrigObj(vector<ObjTrackInfo> &object_list, InputObjInfo *TriggerInfo);
	void drawTrajectory(Mat img_input, Mat &TrackingLine, vector<ObjTrackInfo> &object_list, InputObjInfo *TriggerInfo);
	void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output);
	int Overlap(Rect a, Rect b, double ration);
	double OverlapValue(Rect a, Rect b);
	void BubbleSort(int* array, int size);
	void drawArrow(Mat img, CvPoint p, CvPoint q);
	void object_list_erase(vector<ObjTrackInfo> &object_list, size_t &obj_list_iter);
	void BezierCurve(Point p0, Point p1, Point p2, Point p3, Point *pointArr_output);
	void moveDirect(vector<ObjTrackInfo> &object_list, size_t &obj_list_iter);

private:
	int Max_Mean_Shift_Iter;
	double Similar_Val_Threshold;
	int kernel_type;
	int bin_width;
	int bins;
	Scalar ColorMatrix[10];
	int histSize;
	int count;
	float scaleBetFrame;
	double scaleLearningRate; // scale change rate
	double epsilon;           // min shift in Mean-Shift iteration
	int objNumArray[10];
	int objNumArray_BS[10];
	Scalar *ColorPtr;
	bool suspendUpdate;
	bool addObj;
	bool newObjFind;
	void getKernel(Mat &kernel, const int func_type = 0);
	void computeHist(const Mat &roiMat, const Rect &objBbs, const Mat &kernel, double hist[]);
	int setWeight(const Mat &roiMat, const Rect &objBbs, const Mat &kernel, const double tarHist[], const double candHist[], Mat &weight);
	bool testObjectIntersection(ObjTrackInfo &obj1, ObjTrackInfo &obj2);
	bool testIntraObjectIntersection(vector<ObjTrackInfo> &object_list, int cur_pos);
};

#endif
