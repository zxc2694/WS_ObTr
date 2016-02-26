#ifndef MEANSHIFTTRACKER_H
#define MEANSHIFTTRACKER_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

/* Tracking parameters */
#define stopTrackingObjWithTooSmallWidth_Scale 120 // Delete too small tracking obj when its width becomes < (imgWidth + imgHeight) / stopTrackingObjWithTooSmallWidth_Scale
#define stopTrackingObjWithTooSmallHeight_Scale 28 // Delete too small tracking obj when its height becomes < (imgWidth + imgHeight) / stopTrackingObjWithTooSmallHeight_Scale
#define MAX_DIS_BET_PARTS_OF_ONE_OBJ  38           // Allowed max distance between the box and other tracking boxes
#define MAX_OBJ_LIST_SIZE            100           // Allowed max number of objects 
#define Pixel32S(img,x,y) ((int*)img.data)[(y)*img.cols + (x)] // Get two original tracking boxes'distance
#define minObjWidth_Ini_Scale  60                  // if obj bbs found by bbsFinder has width < (imgWidth + imgHeight) / minObjWidth_Ini_Scale, then addTrackedList don't add it into object_list to track it
#define minObjHeight_Ini_Scale 14                  // if obj bbs found by bbsFinder has height < (imgWidth + imgHeight) / minObjHeight_Ini, then addTrackedList don't add it into object_list to track it

/* Display */
#define plotLineLength          99  // Set tracking line length, (allowed range: 0~99)
#define DELE_RECT_FRAMENO        2  // Allowed frames for boxes of loiter (suggest range: 5~15)
#define occSolve                 2  // 0: not use, 1: use color hist, 2:directly exchange
#define keepTrajectory           0  // 0: not keep, 1: keep. (by color hist)
#define setPointY                4  // Proportional position. 0: Top of the head, 10: Soles of the feet (Range:0~10)
#define display_kalmanRectangle  0  // 0: Not show KF rectangles, 1: Show KF rectangles
#define display_kalmanArrow      0  // 0: Not show KF arrows, 1: Show KF arrows 
#define demoMode                 1  // Without accumulating number (0:debug mode, 1:demo mode) 

/* Math */
#define PI 3.141592653589793238463 

const short MaxHistBins = 4096;

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
	Size	objSize;
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
	int position;                 //1:up, 2:down, 3:left, 4:right
} ObjTrackInfo;

typedef struct
{
	Rect	boundingBox;
	bool	bIsTrigger;
} InputObjInfo;

typedef struct
{
	float value;
	int	objNum;
} OverlapCompare;


class MeanShiftTracker
{
public:
	MeanShiftTracker(int imgWidth, int imgHeight, int MinObjWidth_Ini_Scale, int MinObjHeight_Ini_Scale, int StopTrackingObjWithTooSmallWidth_Scale, int StopTrackingObjWithTooSmallHeight_Scale);
	~MeanShiftTracker();

	Mat DistMat;
	int DistBetObj(Rect a, Rect b);
	void addTrackedList(const Mat &img, vector<ObjTrackInfo> &object_list, Rect bbs, short type);
	void updateObjBbs(const Mat &img, vector<ObjTrackInfo> &object_list, Rect bbs, int idx);
	bool checkTrackedList(vector<ObjTrackInfo> &object_list, vector<ObjTrackInfo> &prev_object_list);
	bool updateTrackedList(vector<ObjTrackInfo> &object_list, vector<ObjTrackInfo> &prev_object_list);
	void drawTrackBox(Mat &img, vector<ObjTrackInfo> &object_list);
	void drawTrackTrajectory(Mat &TrackingLine, vector<ObjTrackInfo> &object_list, size_t &obj_list_iter);
	int track(Mat &img, vector<ObjTrackInfo> &object_list);
	void modifyTrackBox(Mat img_input, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list, CvRect *bbs, int MaxObjNum);
	void occlusionNewObj(Mat img_input, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list, CvRect *bbs, int MaxObjNum);

private:
	// don't tracking too small obj 
	int minObjWidth_Ini;
	int minObjHeight_Ini;
	//const int minObjArea_Ini = IMG_WIDTH*IMG_HEIGHT / 30;

	// del too small obj 
	int minObjWidth;
	int minObjHeight;
	//const int minObjArea = 1000;

	int Max_Mean_Shift_Iter;
	double Similar_Val_Threshold;
	int  kernel_type;
	int	 bin_width;
	int  bins;
	
	Scalar ColorMatrix[10];
	int histSize;
	int count;
	float scaleBetFrame;
	double scaleLearningRate; // scale change rate
	double epsilon;           // min shift in Mean-Shift iteration
	void getKernel(Mat &kernel, const int func_type = 0);
	void computeHist(const Mat &roiMat, const Mat &kernel, double hist[]);
	int setWeight(const Mat &roiMat, const Mat &kernel, const double tarHist[], const double candHist[], Mat &weight);
	bool testObjectIntersection(ObjTrackInfo &obj1, ObjTrackInfo &obj2);
	bool testIntraObjectIntersection(vector<ObjTrackInfo> &object_list, int cur_pos);
};

class KalmanF
{
public:
	KalmanF()
	{
		ticks = 0;
		found = false;
		notFoundCount = 0;
		precTick = ticks;
		stateSize = 6;
		measSize = 4;
		contrSize = 0;
		type = CV_32F;

		kf[0] = KalmanFilter(stateSize, measSize, contrSize, type); state[0] = Mat(stateSize, 1, type);	meas[0] = Mat(measSize, 1, type);
		kf[1] = KalmanFilter(stateSize, measSize, contrSize, type);	state[1] = Mat(stateSize, 1, type);	meas[1] = Mat(measSize, 1, type);
		kf[2] = KalmanFilter(stateSize, measSize, contrSize, type);	state[2] = Mat(stateSize, 1, type);	meas[2] = Mat(measSize, 1, type);
		kf[3] = KalmanFilter(stateSize, measSize, contrSize, type);	state[3] = Mat(stateSize, 1, type);	meas[3] = Mat(measSize, 1, type);
		kf[4] = KalmanFilter(stateSize, measSize, contrSize, type);	state[4] = Mat(stateSize, 1, type);	meas[4] = Mat(measSize, 1, type);
		kf[5] = KalmanFilter(stateSize, measSize, contrSize, type);	state[5] = Mat(stateSize, 1, type);	meas[5] = Mat(measSize, 1, type);
		kf[6] = KalmanFilter(stateSize, measSize, contrSize, type);	state[6] = Mat(stateSize, 1, type);	meas[6] = Mat(measSize, 1, type);
		kf[7] = KalmanFilter(stateSize, measSize, contrSize, type);	state[7] = Mat(stateSize, 1, type);	meas[7] = Mat(measSize, 1, type);
		kf[8] = KalmanFilter(stateSize, measSize, contrSize, type);	state[8] = Mat(stateSize, 1, type);	meas[8] = Mat(measSize, 1, type);
		kf[9] = KalmanFilter(stateSize, measSize, contrSize, type);	state[9] = Mat(stateSize, 1, type);	meas[9] = Mat(measSize, 1, type);
	}
	~KalmanF(){}
	void Init();
	void Predict(vector<ObjTrackInfo> &object_list, vector<cv::Rect> &ballsBox);
	void Update(vector<ObjTrackInfo> &object_list, vector<cv::Rect> &ballsBox, int Upate);
	void drawPredBox(Mat &img);
	double ticks;
	bool found;
	int notFoundCount;
	double precTick;
	double dT;

private:
	int stateSize;
	int measSize;
	int contrSize;
	unsigned int type;
	KalmanFilter kf[10];
	Mat state[10];
	Mat meas[10];
	Rect predRect[10];
	Point center[10];
	int pred_x[10];
	int pred_y[10];
};

void tracking_function(Mat &img_input, Mat &img_output, CvRect *bbs, int MaxObjNum, InputObjInfo *trigROI);
void revertBbsSize(Mat &img_input, CvRect *bbs, int &MaxObjNum);
void ObjNumArr(int *objNumArray, int *objNumArray_BS);
void getNewObj(Mat img_input, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list, CvRect *bbs, int MaxObjNum);
void findTrigObj(vector<ObjTrackInfo> &object_list, InputObjInfo *TriggerInfo);
void drawTrajectory(Mat img_input, Mat &TrackingLine, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list, InputObjInfo *TriggerInfo);
void KFtrack(Mat &img_input, vector<ObjTrackInfo> &object_list, KalmanF &KF);
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location);
int Overlap(Rect a, Rect b, double ration);
double OverlapValue(Rect a, Rect b);
void BubbleSort(int* array, int size);
void KF_init(cv::KalmanFilter *kf);
void ComparePoint_9(IplImage fgmaskIpl, vector<ObjTrackInfo> &object_list, int obj_list_iter, int PtN);
void drawArrow(Mat img, CvPoint p, CvPoint q);
void object_list_erase(vector<ObjTrackInfo> &object_list, size_t &obj_list_iter);


#endif