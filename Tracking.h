#ifndef MEANSHIFTTRACKER_H
#define MEANSHIFTTRACKER_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
//#include "MotionDetection.h"

using namespace cv;
using namespace std;

const short MaxHistBins = 4096;

/* Set tracking line length, range: 20~100 */
#define plotLineLength   99

/* Setting 1 if you want to display it */
#define display_bbsRectangle     0
#define display_kalmanRectangle  0
#define display_kalmanArrow      0

/* BBS parameters */
#define connectedComponentPerimeterScale 6.0f      // when compute obj bbs, ignore obj with perimeter < (imgWidth + imgHeight) / (imgCompressionScale * ConnectedComponentPerimeterScale)
#define minObjWidth_Ini_Scale  60                  // if obj bbs found by bbsFinder has width < (imgWidth + imgHeight) / minObjWidth_Ini_Scale, then addTrackedList don't add it into object_list to track it
#define minObjHeight_Ini_Scale 14                  // if obj bbs found by bbsFinder has height < (imgWidth + imgHeight) / minObjHeight_Ini, then addTrackedList don't add it into object_list to track it

/* del too small obj from object_list (ie give up tracking it) */
#define stopTrackingObjWithTooSmallWidth_Scale 120 // stop tracking obj when its width becomes < (imgWidth + imgHeight) / stopTrackingObjWithTooSmallWidth_Scale
#define stopTrackingObjWithTooSmallHeight_Scale 28 // stop tracking obj when its height becomes < (imgWidth + imgHeight) / stopTrackingObjWithTooSmallHeight_Scale

#define Pixel32S(img,x,y) ((int*)img.data)[(y)*img.cols + (x)]
#define CVCONTOUR_APPROX_LEVEL         2      
#define CVCLOSE_ITR                    3	
#define MAX_DIS_BET_PARTS_OF_ONE_OBJ  38
#define MAX_OBJ_LIST_SIZE            100
#define PI       3.141592653589793238463
#define DELE_RECT_FRAMENO              1

/* Debug or demo */
#define demoMode  1 // Without accumulating number and saving images output (0:debug, 1:demo) 


class FindConnectedComponents
{
public:
	FindConnectedComponents(int imgWidth, int imgHeight, int ImgCompressionScale, float ConnectedComponentPerimeterScale)
	{
		MaxObjNum = 10;         // bbsFinder don't find more than MaxObjNum objects  
		method_Poly1_Hull0 = 1; // Use Polygon algorithm if method_Poly1_Hull0 = 1, and use Hull algorithm if method_Poly1_Hull0 = 0
		minConnectedComponentPerimeter = (imgWidth + imgHeight) / (ImgCompressionScale * ConnectedComponentPerimeterScale);
	}
	~FindConnectedComponents(){}

	void returnBbs(Mat BS_input, int *num, CvRect *bbs, CvPoint *centers, bool ignoreTooSmallPerimeter);
	void shadowRemove(Mat BS_input, int *num, CvRect *bbs, CvPoint *centers);
private:
	int method_Poly1_Hull0;
	int minConnectedComponentPerimeter; // ignores obj with too small perimeter 
	int MaxObjNum;
};

typedef struct
{
	int No;						// numbers of track boxes 
	short	type;				// 1:vehicle , 2: pedestrian, 3: unknown
	short	status;				// 1: detected, 2: tracked, 3: miss to detect, 4: loss to track
	//Point   bbsCen;             // tracked obj's bbs's center   
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
	int ComparePoint[10][DELE_RECT_FRAMENO]; //It decides whether rectangles is motionless or not.

} Object2D;

typedef struct
{
	short	type;				// 1: vehicle, 2:pedestrian , 3: unknown
	short	status;				// 1: detected, 2: tracked, 3: miss to detect, 4: loss to track
	Rect	boundingBox;		// in pixels
	Point3f xyz0;				// 3d position of previous time instance in world coordinate, minimum distance from camera
	Point3f xyz;				// 3d position of currnet time in world coordinate, minimum distance from camera
	double	hist[MaxHistBins];	// disparity(32_bins) + intensity(32_bins) : for tracking
	vector<double> descriptor;
} Object3D;

class IObjectTracker
{
public:
	IObjectTracker(){ count = 0; }
	~IObjectTracker(){}

	//virtual void addTrackedList(const Mat &img, vector<Object2D> &object_list, Object2D &obj) = 0;
	Mat DistMat;
	Scalar ColorMatrix[10];
	int histSize;

	virtual int DistBetObj(Rect a, Rect b) = 0;
	virtual void addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type) = 0;
	virtual void updateObjBbs(const Mat &img, vector<Object2D> &object_list, Rect bbs, int idx) = 0;
	virtual int  track(Mat &img, vector<Object2D> &object_list) = 0; // track single object

	virtual bool checkTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list) = 0;
	virtual bool updateTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list) = 0;
	virtual void drawTrackBox(Mat &img, vector<Object2D> &object_list) = 0;
	virtual void  drawTrackTrajectory(Mat &TrackingLine, vector<Object2D> &object_list, size_t &obj_list_iter) = 0;
	//double getDistanceThreshold(){ return Dist_Threshold; }

	//double track(Mat &img, Object2D &object); // track single object
	//double getDistanceThreshold(){ return Dist_Threshold; }
	//vector<double> track2(Mat &img, vector<Object2D> &object_list); // track multiple objects
	int count;

private:
	/*DescriptorFactory *pDescriptorFac;
	IObjectDescriptor *HOG;
	const double	Dist_Threshold = 0.1f;*/

	//double computeDistance(vector<double> &feature1, vector<double> &feature2);
};

//IObjectTracker::IObjectTracker()
//{
//	pDescriptorFac = new DescriptorFactory();
//	HOG = pDescriptorFac->create("HOG");
//}
//
//IObjectTracker::~IObjectTracker()
//{
//	delete pDescriptorFac;
//}

//double IObjectTracker::track(Mat &img, Object2D &object)
//{
//	if (img.data == NULL) return -1;
//	//if (object.size() == 0) return -1;
//	Rect search_area;
//	search_area.x = max(0.0f, object.boundingBox.x - 0.7f * object.boundingBox.width);
//	search_area.y = max(0.0f, object.boundingBox.y - 0.7f * object.boundingBox.height);
//	search_area.width = (object.boundingBox.x + 2.1 * object.boundingBox.width) >= img.cols ? img.cols - object.boundingBox.x - 1 : 2.1 * object.boundingBox.width;
//	search_area.height = (object.boundingBox.y + 2.1 * object.boundingBox.height) >= img.rows ? img.rows - object.boundingBox.y - 1 : 2.1 * object.boundingBox.height;
//	vector<double> descriptor;
//	double MinDist = 10000;
//	double Magnit_Thre = 55000;
//	for (int y = search_area.y; y + object.boundingBox.height < search_area.y + search_area.height; y += 3){
//		for (int x = search_area.x; x + object.boundingBox.width < search_area.x + search_area.width; x += 3){
//			Rect roi = Rect(x, y, object.boundingBox.width, object.boundingBox.height);
//			Mat roi_img = img(roi);
//			cv::resize(roi_img, roi_img, cv::Size(66, 130));
//			double magnit = HOG->computeDescriptor(roi_img, descriptor);
//			if (magnit < Magnit_Thre)
//				continue;
//			double dist = computeDistance(object.descriptor, descriptor);
//			if (dist < MinDist){
//				MinDist = dist;
//				object.boundingBox = roi;
//				object.status = 2; // tracked
//			}
//		}
//	}
//	return MinDist;
//}
//
//double IObjectTracker::computeDistance(vector<double> &feature1, vector<double> &feature2)
//{
//	double sum = 0;
//	for (size_t i = 0; i < feature1.size(); i++){
//		sum += fabs(feature1[i] - feature2[i]);
//	}
//	return sum / feature1.size();
//}

//vector<double> IObjectTracker::track2(Mat &img, vector<Object2D> &object_list)
//{
//	vector<double> track_prob;
//	if (img.data == NULL) return track_prob;
//	for (size_t c = 0; c < object_list.size(); c++){
//		Rect search_area;
//		search_area.x = max(0.0f, object_list[c].boundingBox.x - 1.0f * object_list[c].boundingBox.width);
//		search_area.y = max(0.0f, object_list[c].boundingBox.y - 1.0f * object_list[c].boundingBox.height);
//		search_area.width = (object_list[c].boundingBox.x + 3 * object_list[c].boundingBox.width) >= img.cols ? img.cols - object_list[c].boundingBox.x - 1 : 4 * object_list[c].boundingBox.width;
//		search_area.height = (object_list[c].boundingBox.y + 3 * object_list[c].boundingBox.height) >= img.rows ? img.rows - object_list[c].boundingBox.y - 1 : 4 * object_list[c].boundingBox.height;
//		vector<double> descriptor;
//		double MinDist = 10000;
//		double Magnit_Thre = 50000;
//		for (int y = search_area.y; y + object_list[c].boundingBox.height < search_area.y + search_area.height; y += 4){
//			for (int x = search_area.x; x + object_list[c].boundingBox.width < search_area.x + search_area.width; x += 4){
//				Rect roi = Rect(x, y, object_list[c].boundingBox.width, object_list[c].boundingBox.height);
//				Mat roi_img = img(roi);
//				cv::resize(roi_img, roi_img, cv::Size(66, 130));
//				double magnit = HOG->computeDescriptor(roi_img, descriptor);
//				if (magnit < Magnit_Thre)
//					continue;
//				double dist = computeDistance(object_list[c].descriptor, descriptor);
//				if (dist < MinDist){
//					MinDist = dist;
//					object_list[c].boundingBox = roi;
//				}
//			}
//		}
//	}
//}

class MeanShiftTracker : public IObjectTracker
{
public:
	MeanShiftTracker(int imgWidth, int imgHeight, int MinObjWidth_Ini_Scale, int MinObjHeight_Ini_Scale, int StopTrackingObjWithTooSmallWidth_Scale, int StopTrackingObjWithTooSmallHeight_Scale);
	~MeanShiftTracker();

	int DistBetObj(Rect a, Rect b);
	void addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type);
	void updateObjBbs(const Mat &img, vector<Object2D> &object_list, Rect bbs, int idx);
	bool checkTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list);
	bool updateTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list);
	void drawTrackBox(Mat &img, vector<Object2D> &object_list);
	void  drawTrackTrajectory(Mat &TrackingLine, vector<Object2D> &object_list, size_t &obj_list_iter);
	int  track(Mat &img, vector<Object2D> &object_list);
	int count;

private:
	// don't tracking too small obj 
    int minObjWidth_Ini;
    int minObjHeight_Ini;
	//const int minObjArea_Ini = IMG_WIDTH*IMG_HEIGHT / 30;

	// del too small obj 
    int minObjWidth;
    int minObjHeight;
	//const int minObjArea = 1000;

	const int Max_Mean_Shift_Iter = 8;
	const double Similar_Val_Threshold = 0.165;
	int  kernel_type;
	int	 bin_width;
	int  bins;
	const float scaleBetFrame = 0.1;
	const double scaleLearningRate = 0.1; // scale change rate
	const double epsilon = 1; // min shift in Mean-Shift iteration
	void getKernel(Mat &kernel, const int func_type = 0);
	void computeHist(const Mat &roiMat, const Mat &kernel, double hist[]);
	int setWeight(const Mat &roiMat, const Mat &kernel, const double tarHist[], const double candHist[], Mat &weight);
	bool testObjectIntersection(Object2D &obj1, Object2D &obj2);
	bool testIntraObjectIntersection(vector<Object2D> &object_list, int cur_pos);
};

class KalmanF
{
public:
	KalmanF()
	{
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
	void Predict(vector<Object2D> &object_list, vector<cv::Rect> &ballsBox);
	void Update(vector<Object2D> &object_list, vector<cv::Rect> &ballsBox, int Upate);
	void drawPredBox(Mat &img);
	double ticks = 0;
	bool found = false;
	int notFoundCount = 0;
	double precTick = ticks;
	double dT;

private:
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
	unsigned int type = CV_32F;
	KalmanFilter kf[10];
	Mat state[10];
	Mat meas[10];
	Rect predRect[10];
	Point center[10];
	int pred_x[10];
	int pred_y[10];
};

void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location);
int Overlap(Rect a, Rect b, double ration);
void MorphologyProcess(IplImage* &fgmaskIpl);
void BubbleSort(int* array, int size);
void tracking_function(Mat &img, Mat &fgmask, int &nframes, CvRect *bbs, int MaxObjNum, int Mode);
void KF_init(cv::KalmanFilter *kf);
void ComparePoint_9(IplImage *fgmaskIpl, vector<Object2D> &object_list, int obj_list_iter, int PtN);
void drawArrow(Mat img, CvPoint p, CvPoint q);
int FindObjBlackPoints(vector<Object2D> &object_list, int obj_list_iter);

#endif