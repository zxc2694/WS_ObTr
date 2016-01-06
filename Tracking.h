#ifndef MEANSHIFTTRACKER_H
#define MEANSHIFTTRACKER_H
#define MEANSHIFTTRACKER_H

#include <memory>
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

const short MaxHistBins = 4096;

/* Update the initial frame number of codebook */
#define nframesToLearnBG  1  //if you use codebook, set 300. If you use MOG, set 1

/* Set tracking line length, range: 20~100 */
#define plotLineLength   30

#define Pixel32S(img,x,y) ((int*)img.data)[(y)*img.cols + (x)]
#define max(X, Y) (((X) >= (Y)) ? (X) : (Y))
#define min(X, Y) (((X) <= (Y)) ? (X) : (Y))

#define CVCONTOUR_APPROX_LEVEL         2
#define CVCLOSE_ITR                    1	
#define MAX_DIS_BET_PARTS_OF_ONE_OBJ  38
#define MAX_OBJ_LIST_SIZE            100

typedef struct
{
	int No;						// numbers of track boxes 
	short	type;				// 1:vehicle , 2: pedestrian, 3: unknown
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
	int countDone;
	int times;
} Object2D;

typedef struct
{
	short	type;				// 1: vehicle, 2:pedestrian , 3: unknown
	short	status;				// 1: detected, 2: tracked, 3: miss to detect, 4: loss to track
	Rect	boundingBox;		// in pixels
	Point3f xyz0;				// 3d position of previous time instance in world coordinate, minimum distance from camera
	Point3f xyz;				// 3d position of currnet time in world coordinate, minimum distance from camera
	float	hist[MaxHistBins];	// disparity(32_bins) + intensity(32_bins) : for tracking
	vector<float> descriptor;
} Object3D;

class IObjectTracker
{
public:
	IObjectTracker(){}
	~IObjectTracker(){}


	//virtual void addTrackedList(const Mat &img, vector<Object2D> &object_list, Object2D &obj) = 0;
	Mat DistMat;
	Scalar ColorMatrix[10];

	virtual int DistBetObj(Rect a, Rect b) = 0;
	virtual void addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type) = 0;
	virtual void updateObjBbs(const Mat &img, vector<Object2D> &object_list, Rect bbs, int idx) = 0;
	virtual int  track(Mat &img, vector<Object2D> &object_list) = 0; // track single object

	virtual bool checkTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list) = 0;
	virtual bool updateTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list) = 0;
	virtual void drawTrackBox(Mat &img, vector<Object2D> &object_list) = 0;
	virtual int  drawTrackTrajectory(Mat &TrackingLine, vector<Object2D> &object_list, size_t &obj_list_iter) = 0;
	//float getDistanceThreshold(){ return Dist_Threshold; }

	//float track(Mat &img, Object2D &object); // track single object
	//float getDistanceThreshold(){ return Dist_Threshold; }
	//vector<float> track2(Mat &img, vector<Object2D> &object_list); // track multiple objects
private:
	int count = 0;
	/*DescriptorFactory *pDescriptorFac;
	IObjectDescriptor *HOG;
	const float	Dist_Threshold = 0.1f;*/

	//float computeDistance(vector<float> &feature1, vector<float> &feature2);

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

//float IObjectTracker::track(Mat &img, Object2D &object)
//{
//	if (img.data == NULL) return -1;
//	//if (object.size() == 0) return -1;
//	Rect search_area;
//	search_area.x = max(0.0f, object.boundingBox.x - 0.7f * object.boundingBox.width);
//	search_area.y = max(0.0f, object.boundingBox.y - 0.7f * object.boundingBox.height);
//	search_area.width = (object.boundingBox.x + 2.1 * object.boundingBox.width) >= img.cols ? img.cols - object.boundingBox.x - 1 : 2.1 * object.boundingBox.width;
//	search_area.height = (object.boundingBox.y + 2.1 * object.boundingBox.height) >= img.rows ? img.rows - object.boundingBox.y - 1 : 2.1 * object.boundingBox.height;
//	vector<float> descriptor;
//	float MinDist = 10000;
//	double Magnit_Thre = 55000;
//	for (int y = search_area.y; y + object.boundingBox.height < search_area.y + search_area.height; y += 3){
//		for (int x = search_area.x; x + object.boundingBox.width < search_area.x + search_area.width; x += 3){
//			Rect roi = Rect(x, y, object.boundingBox.width, object.boundingBox.height);
//			Mat roi_img = img(roi);
//			cv::resize(roi_img, roi_img, cv::Size(66, 130));
//			double magnit = HOG->computeDescriptor(roi_img, descriptor);
//			if (magnit < Magnit_Thre)
//				continue;
//			float dist = computeDistance(object.descriptor, descriptor);
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
//float IObjectTracker::computeDistance(vector<float> &feature1, vector<float> &feature2)
//{
//	float sum = 0;
//	for (size_t i = 0; i < feature1.size(); i++){
//		sum += fabs(feature1[i] - feature2[i]);
//	}
//	return sum / feature1.size();
//}

//vector<float> IObjectTracker::track2(Mat &img, vector<Object2D> &object_list)
//{
//	vector<float> track_prob;
//	if (img.data == NULL) return track_prob;
//	for (size_t c = 0; c < object_list.size(); c++){
//		Rect search_area;
//		search_area.x = max(0.0f, object_list[c].boundingBox.x - 1.0f * object_list[c].boundingBox.width);
//		search_area.y = max(0.0f, object_list[c].boundingBox.y - 1.0f * object_list[c].boundingBox.height);
//		search_area.width = (object_list[c].boundingBox.x + 3 * object_list[c].boundingBox.width) >= img.cols ? img.cols - object_list[c].boundingBox.x - 1 : 4 * object_list[c].boundingBox.width;
//		search_area.height = (object_list[c].boundingBox.y + 3 * object_list[c].boundingBox.height) >= img.rows ? img.rows - object_list[c].boundingBox.y - 1 : 4 * object_list[c].boundingBox.height;
//		vector<float> descriptor;
//		float MinDist = 10000;
//		double Magnit_Thre = 50000;
//		for (int y = search_area.y; y + object_list[c].boundingBox.height < search_area.y + search_area.height; y += 4){
//			for (int x = search_area.x; x + object_list[c].boundingBox.width < search_area.x + search_area.width; x += 4){
//				Rect roi = Rect(x, y, object_list[c].boundingBox.width, object_list[c].boundingBox.height);
//				Mat roi_img = img(roi);
//				cv::resize(roi_img, roi_img, cv::Size(66, 130));
//				double magnit = HOG->computeDescriptor(roi_img, descriptor);
//				if (magnit < Magnit_Thre)
//					continue;
//				float dist = computeDistance(object_list[c].descriptor, descriptor);
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
	MeanShiftTracker();
	~MeanShiftTracker();

	int DistBetObj(Rect a, Rect b);
	void addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type);
	void updateObjBbs(const Mat &img, vector<Object2D> &object_list, Rect bbs, int idx);
	bool checkTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list);
	bool updateTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list);
	void drawTrackBox(Mat &img, vector<Object2D> &object_list);
	int  drawTrackTrajectory(Mat &TrackingLine, vector<Object2D> &object_list, size_t &obj_list_iter);
	int  track(Mat &img, vector<Object2D> &object_list);
	//
	void setRadius(int _r){ radius = _r; }
	void setBinWidt(int _bin_width){ bin_width = _bin_width; }
private:


	int count = 0;
	const int Max_Iters = 8;
	const float Similar_Val_Threshold = 0.165;
	int  kernel_type;
	int  radius;
	int	 bin_width;
	int  bins;
	void getParzenWindow(Mat &kernel, const int R, const int func_type = 0);
	void getDensityEstimate(const Mat &roiMat, const Mat &kernel, float hist[]);
	double computeSimilarity(const Mat &roiMat, const Mat &kernel, const float target[], const float candidate[], Mat &weight);
	//
	bool testObjectIntersection(Object2D &obj1, Object2D &obj2);
	bool testIntraObjectIntersection(vector<Object2D> &object_list, int cur_pos);
};

void CodeBookInit();
void RunCodeBook(IplImage* &image, IplImage* &yuvImage, IplImage* &ImaskCodeBook, IplImage* &ImaskCodeBookCC, int &nframes);
void find_connected_components(IplImage *mask, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers);
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location);
int Overlap(Rect a, Rect b, float ration);
void MorphologyProcess(IplImage* &fgmaskIpl);
void BubbleSort(int* array, int size);
void tracking_function(Mat &img, Mat &fgmask, IObjectTracker *ms_tracker, int &nframes);
void tracking_ROI_function(Mat &img, CvRect *bbs, int &MaxObjNum, IObjectTracker *ms_tracker, int &nframes);

#endif