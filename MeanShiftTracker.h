#ifndef MEANSHIFTTRACKER_H
#define MEANSHIFTTRACKER_H
#define MEANSHIFTTRACKER_H
#define MAX_OBJ_LIST_SIZE    100
#define Pixel32S(img,x,y)	((int*)img.data)[(y)*img.cols + (x)]
#include "opencv2/core/core.hpp"
#include "ObjectTracker.h"

using namespace cv;

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
#endif