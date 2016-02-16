#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <iostream>
#include <iomanip> 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include "Tracking.h"

int objNumArray[10];
int objNumArray_BS[10];
Scalar *ColorPtr;

void tracking_function(Mat &img_input, Mat &img_output, CvRect *bbs, int MaxObjNum, InputObjInfo &trigROI)
{
	static char runFirst = true;
	static vector<Object2D> object_list;
	static KalmanF KF;
	static IObjectTracker *ms_tracker = new MeanShiftTracker(img_input.cols, img_input.rows, minObjWidth_Ini_Scale, minObjHeight_Ini_Scale, stopTrackingObjWithTooSmallWidth_Scale, stopTrackingObjWithTooSmallHeight_Scale);
	static Mat TrackingLine(img_input.rows, img_input.cols, CV_8UC4);
	TrackingLine = Scalar::all(0);

	// Kalman Filter initialization
	if (runFirst)  KF.Init();

	// Enlarge the size of bbs 2 times
	revertBbsSize(img_input, bbs, MaxObjNum); 
	
	// Arrange object number to prevent accumulation
	ObjNumArr(objNumArray, objNumArray_BS);

	// Main tracking code using Mean-shift algorithm
	ms_tracker->track(img_input, object_list);         	

	// Add new useful ROI to the object_list for tracking
	getNewObj(img_input, ms_tracker, object_list, bbs, MaxObjNum);			

	// Modify the size of the tracking boxes and delete useless boxes
	modifyTrackBox(img_input, ms_tracker, object_list, bbs, MaxObjNum);

	// Find trigger object
	findTrigObj(object_list, trigROI);

	// Draw all the track boxes and their numbers 
	ms_tracker->drawTrackBox(img_input, object_list);

	//Plotting trajectories
	drawTrajectory(img_input, TrackingLine, ms_tracker, object_list, trigROI);

	// Prediction and update of Kalman Filter 	
	KFtrack(img_input, object_list, KF);	

	// Tracking image output (merge 3-channel image and 4-channel trakcing lines)
	overlayImage(img_input, TrackingLine, img_output, cv::Point(0, 0));

	runFirst = false;
}

void revertBbsSize(Mat &img_input, CvRect *bbs, int &MaxObjNum)
{
	for (int iter = 0; iter < MaxObjNum; ++iter)
	{
		bbs[iter].x *= 2;
		bbs[iter].y *= 2;
		bbs[iter].width *= 2;
		bbs[iter].height *= 2;
	}
	if (display_bbsRectangle == true)
	{
		/* Plot the rectangles background subtarction finds */
		for (int iter = 0; iter < MaxObjNum; iter++){
			rectangle(img_input, bbs[iter], Scalar(0, 255, 255), 2);
		}
	}
}

void ObjNumArr(int *objNumArray, int *objNumArray_BS)
{
	static char runFirst = true;
	if (runFirst)
	{
		for (unsigned int s = 0; s < 10; s++)
		{
			objNumArray[s] = 65535;
			objNumArray_BS[s] = 65535;
		}
		runFirst = false;
	}
	else 
	{
		for (int iter = 0; iter < 10; iter++)
			objNumArray_BS[iter] = objNumArray[iter];

		BubbleSort(objNumArray_BS, 10);
	}
}

void getNewObj(Mat img_input, IObjectTracker *ms_tracker, vector<Object2D> &object_list, CvRect *bbs, int MaxObjNum)
{
	int bbs_iter;
	size_t obj_list_iter;
	for (bbs_iter = 0; bbs_iter < MaxObjNum; ++bbs_iter)
	{
		bool Overlapping = false, addToList = true;
		vector<int> replaceList;

		for (obj_list_iter = 0; obj_list_iter < object_list.size(); ++obj_list_iter)
		{
			if ((bbs[bbs_iter].width*bbs[bbs_iter].height > 1.8f*object_list[(int)obj_list_iter].boundingBox.width*object_list[(int)obj_list_iter].boundingBox.height)) //If the size of bbs is 1.8 times lagrer than the size of boundingBox, determine whether replace the boundingBox by the following judgement
				// && (bbs[bbs_iter].width*bbs[bbs_iter].height < 4.0f*object_list[obj_list_iter].boundingBox.width*object_list[obj_list_iter].boundingBox.height)
			{
				if (Overlap(bbs[bbs_iter], object_list[(int)obj_list_iter].boundingBox, 0.5f)) // Overlap > 0.5 --> replace the boundingBox
				{
					replaceList.push_back((int)obj_list_iter);
				}
			}
			else
			{
				if (Overlap(bbs[bbs_iter], object_list[(int)obj_list_iter].boundingBox, 0.3f))		addToList = false; // If the size of overlap is small, don't add to object list. (no replace)
			}
		} // end of 2nd for 

		int iter1 = 0, iter2 = 0;

		if ((int)replaceList.size() != 0)
		{

			for (unsigned int iter = 0; iter < object_list.size(); ++iter)
			{
				if ((bbs[bbs_iter].width*bbs[bbs_iter].height <= 1.8f*object_list[iter].boundingBox.width*object_list[iter].boundingBox.height) // contrary to above judgement
					&& Overlap(bbs[bbs_iter], object_list[iter].boundingBox, 0.5f))		replaceList.push_back(iter);
			}

			for (iter1 = 0; iter1 < (int)replaceList.size(); ++iter1)
			{
				for (iter2 = iter1 + 1; iter2 < (int)replaceList.size(); ++iter2)
				{
					/*cout << Pixel32S(ms_tracker->DistMat, MIN(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
					MAX(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No)) << endl;*/

					if (Pixel32S(ms_tracker->DistMat, object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No) > MAX_DIS_BET_PARTS_OF_ONE_OBJ)
					{
						addToList = false;
						goto end;
					}
				}
			}
		}
	end: // break for loop

		if ((int)replaceList.size() != 0 && iter1 == (int)replaceList.size())
		{
			Overlapping = true;

			// choose obj with longest duration from replaceList to update it by new bbs found by codebook
			int  objWithLongestDuration = 0;
			for (int iter = 0; iter < (int)replaceList.size(); ++iter)
			{
				if (object_list[replaceList[iter]].PtCount > object_list[replaceList[objWithLongestDuration]].PtCount)		objWithLongestDuration = iter;
			}

			ms_tracker->updateObjBbs(img_input, object_list, bbs[bbs_iter], replaceList[objWithLongestDuration]);

			replaceList.erase(replaceList.begin() + objWithLongestDuration); // reserve the obj with longest duration in replaceList (exclude it from replaceList)

			if ((int)replaceList.size() > 1)	BubbleSort(&replaceList[0], (int)replaceList.size());

			//for (int iter = 0; iter < (int)replaceList.size(); ++iter)
			for (int iter = (int)replaceList.size() - 1; iter >= 0; --iter)
			{
				for (int iterColor = 0; iterColor < 10; iterColor++)
				{
					if (objNumArray_BS[obj_list_iter] == objNumArray[iterColor])
					{
						objNumArray[iterColor] = 1000; // Recover the value of which the number will be remove  
						break;
					}
				}
				object_list.erase(object_list.begin() + replaceList[iter]);
			}
		}

		if ((!Overlapping && addToList) && ((unsigned int)MaxObjNum > object_list.size()))
		{
			ms_tracker->addTrackedList(img_input, object_list, bbs[bbs_iter], 2); //No replace and add object list -> bbs convert boundingBox.
		}

		vector<int>().swap(replaceList);
	}  // end of 1st for 
}

void modifyTrackBox(Mat img_input, IObjectTracker *ms_tracker, vector<Object2D> &object_list, CvRect *bbs, int MaxObjNum)
{
	/* Modify the size of the tracking box  */
	int bbsNumber = 0;
	for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
	{
		for (int i = 0; i < MaxObjNum; i++)
		{
			// Find how many bbs in the tracking box
			if (Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.5f))
				bbsNumber++;
		}
		// When the width of tracking box has 1.5 times more bigger than the width of bbs:
		for (int i = 0; i < MaxObjNum; i++)
		{
			if ((Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.5f)) && (object_list[obj_list_iter].boundingBox.width > 1.5 * bbs[i].width) && (bbsNumber == 1))
				ms_tracker->updateObjBbs(img_input, object_list, bbs[i], obj_list_iter); //Reset the scale of the tracking box.
		}
	}

	/* Removing motionless tracking box  */
	int black = 0, times = 0;
	for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
	{
		for (int i = 0; i < MaxObjNum; i++)
		{
			if (Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.5f))
				break;
			else
				black++;
		}
		// Restarting count when count > DELE_RECT_FRAMENO number
		if (object_list[obj_list_iter].cPtNumber == DELE_RECT_FRAMENO + 1)
			object_list[obj_list_iter].cPtNumber = 0;

		// findBbs[i] = 0 -> no object; findBbs[i] = 1 -> has object
		if (black == MaxObjNum)
			object_list[obj_list_iter].findBbs[object_list[obj_list_iter].cPtNumber] = 0;
		else
			object_list[obj_list_iter].findBbs[object_list[obj_list_iter].cPtNumber] = 1;

		object_list[obj_list_iter].cPtNumber++;

		// To determine whether it continuously keep no object 
		for (int i = 0; i < DELE_RECT_FRAMENO; i++)
		{
			if (object_list[obj_list_iter].findBbs[i] == 0)
			{
				times++;
			}
		}
		if (DELE_RECT_FRAMENO == times)
		{
			for (int iterColor = 0; iterColor < 10; iterColor++)
			{
				if (objNumArray_BS[obj_list_iter] == objNumArray[iterColor])
				{
					objNumArray[iterColor] = 1000; // Recover the value of which the number will be remove  
					break;
				}
			}
			object_list.erase(object_list.begin() + obj_list_iter); // Remove the tracking box	

			if (object_list.size() == 0)//Prevent out of vector range
				break;

		}
		black = 0;
		times = 0;
	}

	/* Removing one of overlapping tracking box */
	//if (object_list.size() == 2)
	//{
	//	if (Overlap(object_list[0].boundingBox, object_list[1].boundingBox, 0.9f))
	//	{
	//		if (object_list[0].boundingBox.height > object_list[1].boundingBox.height)
	//		{
	//			for (int iterColor = 0; iterColor < 10; iterColor++)
	//			{
	//				if (objNumArray_BS[1] == objNumArray[iterColor])
	//				{
	//					objNumArray[iterColor] = 1000; // Recover the value of which the number will be remove  
	//					break;
	//				}
	//			}
	//			object_list.erase(object_list.begin() + 1); // Remove the tracking box
	//		}
	//		else
	//		{
	//			for (int iterColor = 0; iterColor < 10; iterColor++)
	//			{
	//				if (objNumArray_BS[0] == objNumArray[iterColor])
	//				{
	//					objNumArray[iterColor] = 1000; // Recover the value of which the number will be remove  
	//					break;
	//				}
	//			}
	//			object_list.erase(object_list.begin()); // Remove the tracking box
	//		}
	//	}
	//}
}

void findTrigObj(vector<Object2D> &object_list, InputObjInfo &TriggerInfo)
{
	if (TriggerInfo.bIsTrigger == true)
	{
		for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
		{
			if (Overlap(object_list[obj_list_iter].boundingBox, TriggerInfo.boundingBox, 0.5f))
			{
				object_list[obj_list_iter].bIsDrawing = true;
			}
			else
			{
				object_list[obj_list_iter].bIsDrawing = false;
			}
		}
	}
}

void drawTrajectory(Mat img_input, Mat &TrackingLine, IObjectTracker *ms_tracker, vector<Object2D> &object_list, InputObjInfo &TriggerInfo)
{
	static char prevData = false;
	static int pre_data_X[10], pre_data_Y[10];  

	for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
	{
		if (prevData == true) //prevent plotting tracking line when previous tracking data is none.
		{
			if (TriggerInfo.bIsTrigger == false) // if no trigger, draw all trajectories
			{
				ms_tracker->drawTrackTrajectory(TrackingLine, object_list, obj_list_iter); // Plotting all the tracking lines	
			}
			else // trigger area is being invaded
			{
				if (object_list[obj_list_iter].bIsDrawing == true) // trigger object 
				{
					ms_tracker->drawTrackTrajectory(TrackingLine, object_list, obj_list_iter); // Only plot triggered tracking line	

					drawArrow(img_input,  // Draw the arrow on the pedestrian's head
						Point(0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x),
						(object_list[obj_list_iter].boundingBox.y) - 40)
						, Point(0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x),
						(object_list[obj_list_iter].boundingBox.y) - 20));
				}
			}
		}

		// Get previous point in order to use line function. 
		pre_data_X[obj_list_iter] = (int)(0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x));
		pre_data_Y[obj_list_iter] = (int)(0.9 * object_list[obj_list_iter].boundingBox.height + (object_list[obj_list_iter].boundingBox.y));

		// Restarting count when count > plotLineLength number
		if (object_list[obj_list_iter].PtNumber == plotLineLength + 1)
			object_list[obj_list_iter].PtNumber = 0;

		object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber] = Point(pre_data_X[obj_list_iter], pre_data_Y[obj_list_iter]); //Storage all of points on the array. 

		object_list[obj_list_iter].PtNumber++;
		object_list[obj_list_iter].PtCount++;

	}// end of plotting trajectory
	prevData = true;
}

void KFtrack(Mat &img_input, vector<Object2D> &object_list, KalmanF &KF)
{
	vector<cv::Rect> KFBox;

	/*Kalman Filter Function */
	if ((display_kalmanArrow == true) || (display_kalmanRectangle == true))
	{
		KF.Predict(object_list, KFBox); //Predict bounding box by Kalman filter

		int UpateKF = true;
		KF.drawPredBox(img_input);             // Draw predict bounding boxes	
		if (object_list.size() == 2)
		{
			for (size_t obj_list_iter = 0; obj_list_iter < object_list.size() - 1; obj_list_iter++)
			{
				int p_x = object_list[obj_list_iter].boundingBox.x;
				int p_y = object_list[obj_list_iter].boundingBox.y;
				int q_x = object_list[obj_list_iter + 1].boundingBox.x;
				int q_y = object_list[obj_list_iter + 1].boundingBox.y;
				double hypotenuse = sqrt((p_y - q_y)*(p_y - q_y) + (p_x - q_x)*(p_x - q_x)); //length of pq line
				int intHY = (int)hypotenuse;
				//cout << "hypotenuse" << hypotenuse << endl;

				if (intHY == 64)//if (Overlap(object_list[obj_list_iter].boundingBox, object_list[obj_list_iter + 1].boundingBox, 0.001f))			
					UpateKF = false;
			}
		}
		KF.Update(object_list, KFBox, UpateKF);   // Update of Kalman filter
	}
}

MeanShiftTracker::MeanShiftTracker(int imgWidth, int imgHeight, int MinObjWidth_Ini_Scale, int MinObjHeight_Ini_Scale, int StopTrackingObjWithTooSmallWidth_Scale, int StopTrackingObjWithTooSmallHeight_Scale) : kernel_type(2), bin_width(16), count(0)
{
	// if obj bbs found by bbsFinder is too small, then addTrackedList don't add it into object_list to track it
	minObjWidth_Ini = (imgWidth + imgHeight) / MinObjWidth_Ini_Scale;
	minObjHeight_Ini = (imgWidth + imgHeight) / MinObjHeight_Ini_Scale;
	//const int minObjArea_Ini = IMG_WIDTH*IMG_HEIGHT / 30;

	// del too small obj from object_list (ie stop tracking it)
	minObjWidth = (imgWidth + imgHeight) / StopTrackingObjWithTooSmallWidth_Scale;
	minObjHeight = (imgWidth + imgHeight) / StopTrackingObjWithTooSmallHeight_Scale;
	//const int minObjArea = 1000;

	bins = 256 / bin_width;
	histSize = bins*bins*bins;

	DistMat = Mat::zeros(MAX_OBJ_LIST_SIZE, MAX_OBJ_LIST_SIZE, CV_32SC1);
	ColorMatrix[0] = Scalar(0, 0, 255);
	ColorMatrix[1] = Scalar(255, 0, 0);
	ColorMatrix[2] = Scalar(0, 255, 0);
	ColorMatrix[3] = Scalar(0, 255, 255);
	ColorMatrix[4] = Scalar(255, 0, 255);
	ColorMatrix[5] = Scalar(255, 255, 0);
	ColorMatrix[6] = Scalar(122, 0, 255);
	ColorMatrix[7] = Scalar(255, 122, 0);
	ColorMatrix[8] = Scalar(0, 255, 122);
	ColorMatrix[9] = Scalar(80, 255, 80);

	ColorPtr = &ColorMatrix[0];

	Max_Mean_Shift_Iter = 8;
	Similar_Val_Threshold = 0.165;
	scaleBetFrame = (float)0.1;
	scaleLearningRate = 0.1;
	epsilon = 1;
}

int MeanShiftTracker::DistBetObj(Rect a, Rect b)
{
	int c, d;
	if ((a.x > b.x + b.width || b.x > a.x + a.width) || (a.y > b.y + b.height || b.y > a.y + a.height))
	{
		if (!(a.x > b.x + b.width || b.x > a.x + a.width))		c = 0;
		else	c = MIN(abs(a.x - (b.x + b.width)), abs((a.x + a.width) - b.x));

		if (!(a.y > b.y + b.height || b.y > a.y + a.height))	d = 0;
		else    d = MIN(abs(a.y - (b.y + b.height)), abs((a.y + a.height) - b.y));

		return (int)(sqrt((double)(c*c + d*d)));
	}
	else return 0;
}

void MeanShiftTracker::addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type)
{
	// don't tracking too small obj 
	if (bbs.width < minObjWidth_Ini || bbs.height < minObjHeight_Ini)	return;

	// don't track when obj just emerge at img edge
	if (bbs.x < 3 || bbs.y < 3 || bbs.x + bbs.width > img.cols - 1 || bbs.y + bbs.height > img.rows - 1)		return;

	++count;

	if ((bbs.height & 1) == 0)    bbs.height -= 1; // bbs.height should be odd number
	if ((bbs.width & 1) == 0)    bbs.width -= 1; // bbs.width should be odd number

	Object2D obj;
	obj.No = count;
	//CvScalar Scalar = cvScalar(rand() % 256, rand() % 256, rand() % 256);
	obj.color = ColorMatrix[count % 10];
	obj.type = type;
	obj.status = 3;
	obj.boundingBox = bbs;
	obj.initialBbsWidth = bbs.width;
	obj.initialBbsHeight = bbs.height;
	//obj.bbsCen = Point(bbs.x + (bbs.width - 1) / 2, bbs.y + (bbs.height - 1) / 2);
	obj.xyz.z = 20;
	obj.PtNumber = 0;
	obj.cPtNumber = 0;
	obj.PtCount = 0;
	obj.objScale = 1;
	obj.kernel.create(obj.boundingBox.height, obj.boundingBox.width, CV_64FC1);

	for (int i = 0; i < DELE_RECT_FRAMENO; i++)
		obj.findBbs[i] = 1; // 1: has object; 0: no object -> default: has object

	getKernel(obj.kernel, kernel_type);

	Mat tempMat = img(obj.boundingBox);
	computeHist(tempMat, obj.kernel, obj.hist);

	object_list.push_back(obj);

	for (size_t iter = 0; iter < object_list.size(); ++iter) // Compute distances from new object to the other objects.
	{
		Pixel32S(DistMat, obj.No, object_list[(int)iter].No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);
		Pixel32S(DistMat, object_list[(int)iter].No, obj.No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);
	}

	static int countNo = 0;
	char UpdateNo = false;

	for (size_t iter = 0; iter < 10; ++iter)
	{
		if (objNumArray[iter] == 1000)
		{
			objNumArray[iter] = obj.No;
			UpdateNo = true;
			break;
		}
	}

	if (UpdateNo == false)
	{
		objNumArray[countNo] = obj.No;
		countNo++;
	}
}

void MeanShiftTracker::updateObjBbs(const Mat &img, vector<Object2D> &object_list, Rect bbs, int idx)
{
	if ((bbs.height & 1) == 0)    bbs.height -= 1; // bbs.height should be odd number
	if ((bbs.width & 1) == 0)    bbs.width -= 1; // bbs.width should be odd number
	object_list[idx].objScale = 1;
	object_list[idx].boundingBox = bbs;
	object_list[idx].initialBbsWidth = bbs.width;
	object_list[idx].initialBbsHeight = bbs.height;
	//object_list[idx].bbsCen = Point(bbs.x + (bbs.width - 1) / 2, bbs.y + (bbs.height - 1) / 2);

	object_list[idx].kernel.create(object_list[idx].boundingBox.height, object_list[idx].boundingBox.width, CV_64FC1);
	getKernel(object_list[idx].kernel, kernel_type);

	Mat tempMat = img(object_list[idx].boundingBox);
	computeHist(tempMat, object_list[idx].kernel, object_list[idx].hist);
}

bool MeanShiftTracker::checkTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list)
{
	if (object_list.empty())	return false;

	for (size_t c = 0; c < object_list.size(); c++)
	{
		if (object_list[c].similar_val > Similar_Val_Threshold - 0.02)
		{
			//prev_object_list[c].status = 2;
			//prev_object_list[c].boundingBox = object_list[c].boundingBox;
			//prev_object_list[c].similar_val = object_list[c].similar_val;
		}
		else
		{
			for (int iterColor = 0; iterColor < 10; iterColor++)
			{
				if (objNumArray_BS[c] == objNumArray[iterColor])
				{
					objNumArray[iterColor] = 1000;
					break;
				}
			}
			object_list.erase(object_list.begin() + c);
			//prev_object_list.erase(prev_object_list.begin() + c);
			c--;
		}
	}

	//// hit check: delete element one by one from pre_tracked_list until check finished
	//for (size_t i = 0; i < object_list.size(); i++){
	//	for (size_t j = 0; j < prev_object_list.size(); j++){
	//		/*if (tracked_list[i].type != prev_tracked_list[j].type)
	//		continue;*/
	//		// intersect check
	//		bool bHit = testObjectIntersection(object_list[i], prev_object_list[j]);
	//		// check object distance
	//		//bool bEqualDepth = checkByDepth(tracked_list[i].xyz.z, prev_tracked_list[j].xyz.z);
	//		if (bHit){
	//			prev_object_list.erase(prev_object_list.begin() + j);
	//			if (prev_object_list.empty()) break;
	//			if (j == 0) j = -1;
	//			else j--;
	//			//break;
	//		}
	//	}
	//}
	//// miss detected objects: the left elements in the prev_tracked_list are missed detected objects, and then copy to tracked_list
	//for (size_t c = 0; c < prev_object_list.size(); c++){
	//	prev_object_list[c].status = 3; // missed
	//	object_list.push_back(prev_object_list[c]);
	//}
	//if (prev_object_list.empty()) // no missed object
	//{
	//	prev_object_list = object_list;
	//	return false;
	//}
	//else{
	//	prev_object_list.clear();
	//	prev_object_list = object_list;
	//	return true;
	//}
	return true;
}

bool MeanShiftTracker::updateTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list)
{
	for (size_t c = 0; c < object_list.size(); c++)
	{
		if (object_list[c].status == 3)
		{
			bool bIntraSec = testIntraObjectIntersection(object_list, c);
			if (object_list[c].similar_val > Similar_Val_Threshold - 0.02 && !bIntraSec)
			{
				prev_object_list[c].status = 2;
				prev_object_list[c].boundingBox = object_list[c].boundingBox;
				prev_object_list[c].similar_val = object_list[c].similar_val;
			}
			else
			{
				for (int iterColor = 0; iterColor < 10; iterColor++)
				{
					if (objNumArray_BS[c] == objNumArray[iterColor])
					{
						objNumArray[iterColor] = 1000;
						break;
					}
				}
				object_list.erase(object_list.begin() + c);
				prev_object_list.erase(prev_object_list.begin() + c);
				c--;
			}
		}
	}
	return 1;
}

void MeanShiftTracker::drawTrackBox(Mat &img, vector<Object2D> &object_list)
{
	int iter;
	for (size_t c = 0; c < object_list.size(); c++){
		//for (size_t c = 0; c < 1; c++){
		//if (object_list[c].status == 2){

		if (object_list[c].type == 1) //vehicle	
		{
			std::stringstream ss, ss1, ss2, ss3;
			ss << std::fixed << std::setprecision(2) << object_list[c].xyz.z;
			ss1 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.x;
			ss2 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.y;
			//cv::putText(img, "person:" + ss.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y - 8), 1, 1, ColorMatrix[c]);
			//cv::putText(img, "prob:" + ss1.str() + "," + ss2.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, ColorMatrix[c]);
			//cv::putText(img, "prob:" + ss1.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, ColorMatrix[c]);			
			ss3 << object_list[c].No;
			cv::rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
			cv::putText(img, ss3.str(), Point(object_list[c].boundingBox.x + object_list[c].boundingBox.width / 2 - 10, object_list[c].boundingBox.y + object_list[c].boundingBox.height / 2), 1, 3, object_list[c].color, 3);
		}
		if (object_list[c].type == 2) //pedestrian
		{
			std::stringstream ss, ss1, ss2, ss3;

			if (demoMode == true)
			{
				for (iter = 0; iter < 10; iter++)
				{
					if (objNumArray_BS[c] == objNumArray[iter])
					{
						ss3 << iter + 1;
						break;
					}
				}
				object_list[c].color = *(ColorPtr + iter);
				//cv::rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
			}
			else
			{
				ss3 << object_list[c].No;
				cv::rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
				cv::putText(img, ss3.str(), Point(object_list[c].boundingBox.x + object_list[c].boundingBox.width / 2 - 10, object_list[c].boundingBox.y + object_list[c].boundingBox.height / 2), 1, 3, object_list[c].color, 3);
			}
		}
		//}
	}
}

void MeanShiftTracker::drawTrackTrajectory(Mat &TrackingLine, vector<Object2D> &object_list, size_t &obj_list_iter)
{
	if (object_list[obj_list_iter].PtCount > plotLineLength + 1)										//When plotting arrary is overflow:
	{
		if (object_list[obj_list_iter].PtNumber <= plotLineLength)										// Update of last number will influence plotting line on first number and last number, which must prevent. 
			line(TrackingLine, object_list[obj_list_iter].point[0]													//plotting line on first number and last number				
			, object_list[obj_list_iter].point[plotLineLength], Scalar(object_list[obj_list_iter].color.val[0], object_list[obj_list_iter].color.val[1], object_list[obj_list_iter].color.val[2], 20 + 235 * (plotLineLength - object_list[obj_list_iter].PtNumber) / plotLineLength), 3, 1, 0);

		for (int iter1 = 0; iter1 < object_list[obj_list_iter].PtNumber - 1; iter1++)					//plotting line from first number to PtNumber-1
		{
			line(TrackingLine, object_list[obj_list_iter].point[iter1]
				, object_list[obj_list_iter].point[iter1 + 1], Scalar(object_list[obj_list_iter].color.val[0], object_list[obj_list_iter].color.val[1], object_list[obj_list_iter].color.val[2], 20 + 235 * (plotLineLength - object_list[obj_list_iter].PtNumber + 1 + iter1) / plotLineLength), 3, 1, 0);
		}
		for (int iter2 = 0; iter2 < plotLineLength - object_list[obj_list_iter].PtNumber; iter2++)		//plotting line from PtNumber to last number
		{
			line(TrackingLine, object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber + iter2]
				, object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber + iter2 + 1], Scalar(object_list[obj_list_iter].color.val[0], object_list[obj_list_iter].color.val[1], object_list[obj_list_iter].color.val[2], 20 + 235 * iter2 / plotLineLength), 3, 1, 0);
		}
	}
	else
	{																								//When plotting arrary isn't overflow:
		for (int iter = 1; iter < object_list[obj_list_iter].PtNumber; iter++)
			line(TrackingLine, object_list[obj_list_iter].point[iter - 1]										//Directly plot all the points array storages.
			, object_list[obj_list_iter].point[iter], Scalar(object_list[obj_list_iter].color.val[0], object_list[obj_list_iter].color.val[1], object_list[obj_list_iter].color.val[2], 20 + 235 * (iter - 1) / (object_list[obj_list_iter].PtNumber - 1)), 3, 1, 0);
	}

}


int MeanShiftTracker::track(Mat &img, vector<Object2D> &object_list)
{
	Rect CandBbs[3]; // candidate bbs
	Point CandCen; // candidate bbs center coordinates (let upper left corner of bbs have coordinate (0, 0))
	Mat kernel;
	Mat tempMat, weight;
	double* hist[3];
	double hist0[MaxHistBins], hist1[MaxHistBins], hist2[MaxHistBins];
	hist[0] = hist0;
	hist[1] = hist1;
	hist[2] = hist2;
	double weiSum;
	int Mean_Shift_Iter;
	float scale;
	bool delBbsOutImg;


	for (size_t c = 0; c < object_list.size(); c++)
	{
		//int bestScaleIter;
		double bestScale, similarity, largestSimilarity = 0; // choose best scale as the one with largest similarity to target model 
		bool exceedImgBoundary = true;

		scale = object_list[c].objScale - object_list[c].objScale*scaleBetFrame;
		// for 3 different scales of Candidate Bbs: 1 - scaleBetFrame, 1, 1 + scaleBetFrame
		for (int scaleIter = 0; scaleIter < 3; scale += object_list[c].objScale*scaleBetFrame, ++scaleIter)
		{
			// scale bbs
			if (scaleIter != 1) // down or up scale
			{
				int bbsCen_x = object_list[c].boundingBox.x + (object_list[c].boundingBox.width - 1) / 2;
				int bbsCen_y = object_list[c].boundingBox.y + (object_list[c].boundingBox.height - 1) / 2;
				int halfWidth = (int)((object_list[c].initialBbsWidth - 1) / 2 * scale);
				int halfHeight = (int)((object_list[c].initialBbsHeight - 1) / 2 * scale);
				CandBbs[scaleIter].x = bbsCen_x - halfWidth;
				CandBbs[scaleIter].y = bbsCen_y - halfHeight;
				CandBbs[scaleIter].width = 2 * halfWidth + 1;
				CandBbs[scaleIter].height = 2 * halfHeight + 1;
			}
			else
			{
				CandBbs[scaleIter] == object_list[c].boundingBox;
			}


			// if bbs exceed img boundary after scale, don't scale bbs
			if (CandBbs[scaleIter].x < 0 || CandBbs[scaleIter].y < 0 || CandBbs[scaleIter].br().x >= img.cols || CandBbs[scaleIter].br().y >= img.rows)
			{
				//continue;

				CandBbs[scaleIter] &= Rect(0, 0, img.cols, img.rows); // make bbs be inside img

				if ((CandBbs[scaleIter].height & 1) == 0)    CandBbs[scaleIter].height -= 1; // bbs.height should be odd number
				if ((CandBbs[scaleIter].width & 1) == 0)    CandBbs[scaleIter].width -= 1; // bbs.width should be odd number
			}
			if (CandBbs[scaleIter].width < minObjWidth || CandBbs[scaleIter].height < minObjHeight)   continue; // if bbs is too small after scale, don't scale bbs			


			CandCen = Point((CandBbs[scaleIter].width - 1) / 2, (CandBbs[scaleIter].height - 1) / 2); // let 

			// initialize kernel
			kernel.create(CandBbs[scaleIter].height, CandBbs[scaleIter].width, CV_64FC1);
			getKernel(kernel, kernel_type);

			// compute color hist
			tempMat = img(CandBbs[scaleIter]);
			computeHist(tempMat, kernel, hist[scaleIter]);

			// set weight
			weight.create(CandBbs[scaleIter].height, CandBbs[scaleIter].width, CV_64FC1);
			setWeight(tempMat, kernel, object_list[c].hist, hist[scaleIter], weight);

			Mean_Shift_Iter = 0; // Mean_Shift iteration count
			//Point oldBbsCen = Point(0, 0), newBbsCen = Point(epsilon, 0); // candidate bbs center coordinates during Mean_Shift iteration (let upper left corner of img have coordinate (0, 0))

			// Mean_Shift iteration
			delBbsOutImg = false;
			while (1)
			{
				++Mean_Shift_Iter;

				Point2d normalizedShiftVec = Point2d(0, 0); // computed as sum of w(i,j)*x(i,j) for all pixels (i,j) in bbs, where w(i,j) is weight at (i,j) and x(i,j) is normalized coordinate of (i,j), divided by weiSum
				weiSum = 0; // sum of w(i,j) for all pixels (i,j) in bbs, where w(i,j) is weight at (i,j)

				for (int i = 0; i < CandBbs[scaleIter].height; i++)
				{
					for (int j = 0; j < CandBbs[scaleIter].width; j++)
					{
						if (kernel.at<double>(i, j) == 0)	 continue;

						normalizedShiftVec += (weight.at<double>(i, j)*Point2d((double)(j - CandCen.x) / CandCen.x, (double)(i - CandCen.y) / CandCen.y));
						weiSum += weight.at<double>(i, j);
					}
				}

				normalizedShiftVec.x /= weiSum;
				normalizedShiftVec.y /= weiSum;

				double shift_x = normalizedShiftVec.x * CandCen.x; // denormalized bbs shift in img x-axis
				double shift_y = normalizedShiftVec.y * CandCen.y; // denormalized bbs shift in img y-axis

				CandBbs[scaleIter].x += (int)shift_x;
				CandBbs[scaleIter].y += (int)shift_y;

				// if bbs exceed img boundary after shift, then stop iteration
				if (CandBbs[scaleIter].x < 0 || CandBbs[scaleIter].y < 0 || CandBbs[scaleIter].br().x >= img.cols || CandBbs[scaleIter].br().y >= img.rows)
				{
					CandBbs[scaleIter] &= Rect(0, 0, img.cols, img.rows); // make bbs be inside img

					//break;

					if (CandBbs[scaleIter].width < minObjWidth || CandBbs[scaleIter].height < minObjHeight)
					{
						delBbsOutImg = true;
						break; // if the part of bbs inside img is too small after shift, stop shift 	
					}

					if ((CandBbs[scaleIter].height & 1) == 0)    CandBbs[scaleIter].height -= 1; // bbs.height should be odd number
					if ((CandBbs[scaleIter].width & 1) == 0)    CandBbs[scaleIter].width -= 1; // bbs.width should be odd number

					CandCen = Point((CandBbs[scaleIter].width - 1) / 2, (CandBbs[scaleIter].height - 1) / 2);

					// if bbs has been changed, kernel should also be changed
					kernel.create(CandBbs[scaleIter].height, CandBbs[scaleIter].width, CV_64FC1);
					getKernel(kernel, kernel_type);
					// if bbs has been changed, weight should also be changed
					weight.create(CandBbs[scaleIter].height, CandBbs[scaleIter].width, CV_64FC1);
				}

				// if too small bbs center shift, then stop iteration
				if (pow(shift_x, 2) + pow(shift_y, 2) < epsilon)	break;

				// iterate at most Max_Mean_Shift_Iter times
				if (Mean_Shift_Iter == Max_Mean_Shift_Iter)		break;

				// compute color hist
				tempMat = img(CandBbs[scaleIter]);
				computeHist(tempMat, kernel, hist[scaleIter]);

				// set weight
				setWeight(tempMat, kernel, object_list[c].hist, hist[scaleIter], weight);
			} //end of while

			if (delBbsOutImg)   continue; // if the part of bbs inside img is too small after scale and shift, abandon this scale and choose other scale

			// compute color hist
			tempMat = img(CandBbs[scaleIter]);
			computeHist(tempMat, kernel, hist[scaleIter]);

			// choose scale with largest similarity to target model
			similarity = 0;
			for (int histIdx = 0; histIdx < histSize; ++histIdx) // compute similarity
			{
				similarity += sqrt(object_list[c].hist[histIdx] * hist[scaleIter][histIdx]);
			}
			if (similarity > largestSimilarity) // choose largest similarity
			{
				largestSimilarity = similarity;
				//bestScaleIter = scaleIter;
				bestScale = scale;
			}

			exceedImgBoundary = false;
		} // for all scale

		// if the part of bbs inside img is too small for all scales after shifts, abandon tracking this obj, i.e. delete this obj from object_list 
		if (exceedImgBoundary)
		{
			//for (int iterColor = 0; iterColor < 10; iterColor++)
			//{
			//	if (objNumArray_BS[c] == objNumArray[iterColor])
			//	{
			//		objNumArray[iterColor] = 1000;
			//		break;
			//	}
			//}
			//object_list.erase(object_list.begin() + c);
			//--c;
			//continue;
		}


		// determine scale by bestScale and scaleLearningRate
		object_list[c].objScale = (float)(scaleLearningRate*bestScale + (1 - scaleLearningRate)*object_list[c].objScale);


		// adopt candidate bbs scale determined above and implement Mean-Shift again
		int bbsCen_x = object_list[c].boundingBox.x + (object_list[c].boundingBox.width - 1) / 2;
		int bbsCen_y = object_list[c].boundingBox.y + (object_list[c].boundingBox.height - 1) / 2;
		int halfWidth = (int)((object_list[c].initialBbsWidth - 1) / 2 * object_list[c].objScale);
		int halfHeight = (int)((object_list[c].initialBbsHeight - 1) / 2 * object_list[c].objScale);
		object_list[c].boundingBox.x = bbsCen_x - halfWidth;
		object_list[c].boundingBox.y = bbsCen_y - halfHeight;
		object_list[c].boundingBox.width = 2 * halfWidth + 1;
		object_list[c].boundingBox.height = 2 * halfHeight + 1;

		// if bbs exceed img boundary after scale, don't scale bbs
		if (object_list[c].boundingBox.x < 0 || object_list[c].boundingBox.y < 0 || object_list[c].boundingBox.br().x >= img.cols || object_list[c].boundingBox.br().y >= img.rows)
		{
			object_list[c].boundingBox &= Rect(0, 0, img.cols, img.rows); // make bbs be inside img

			if ((object_list[c].boundingBox.height & 1) == 0)    object_list[c].boundingBox.height -= 1; // bbs.height should be odd number
			if ((object_list[c].boundingBox.width & 1) == 0)    object_list[c].boundingBox.width -= 1; // bbs.width should be odd number
		}

		CandCen = Point((object_list[c].boundingBox.width - 1) / 2, (object_list[c].boundingBox.height - 1) / 2);

		// initialize kernel
		kernel.create(object_list[c].boundingBox.height, object_list[c].boundingBox.width, CV_64FC1);
		getKernel(kernel, kernel_type);

		// compute color hist
		tempMat = img(object_list[c].boundingBox);
		computeHist(tempMat, kernel, hist[0]);

		// set weight
		weight.create(object_list[c].boundingBox.height, object_list[c].boundingBox.width, CV_64FC1);
		setWeight(tempMat, kernel, object_list[c].hist, hist[0], weight);

		Mean_Shift_Iter = 0; // Mean_Shift iteration count

		// Mean_Shift iteration
		delBbsOutImg = false;

		while (1)
		{
			++Mean_Shift_Iter;

			Point2d normalizedShiftVec = Point2d(0, 0); // computed as sum of w(i,j)*x(i,j) for all pixels (i,j) in bbs, where w(i,j) is weight at (i,j) and x(i,j) is normalized coordinate of (i,j), divided by weiSum
			weiSum = 0; // sum of w(i,j) for all pixels (i,j) in bbs, where w(i,j) is weight at (i,j)

			for (int i = 0; i < object_list[c].boundingBox.height; i++)
			{
				for (int j = 0; j < object_list[c].boundingBox.width; j++)
				{
					if (kernel.at<double>(i, j) == 0)	 continue;

					normalizedShiftVec += (weight.at<double>(i, j)*Point2d((double)(j - CandCen.x) / CandCen.x, (double)(i - CandCen.y) / CandCen.y));
					weiSum += weight.at<double>(i, j);
				}
			}

			normalizedShiftVec.x /= weiSum;
			normalizedShiftVec.y /= weiSum;

			double shift_x = normalizedShiftVec.x * CandCen.x; // denormalized shift in img x-axis
			double shift_y = normalizedShiftVec.y * CandCen.y; // denormalized shift in img y-axis
			shift_x = (int)(shift_x + 0.5);
			shift_y = (int)(shift_y + 0.5);

			//if (shift_x > 0)
			//{
			//	shift_x += 0.7;
			//	shift_x = round(shift_x);
			//}
			//else
			//{
			//	shift_x -= 0.7;
			//	shift_x = round(shift_x);
			//}

			//if (shift_y > 0)
			//{
			//	shift_y += 0.7;
			//	shift_y = round(shift_y);
			//}
			//else
			//{
			//	shift_y -= 0.7;
			//	shift_y = round(shift_y);
			//}

			object_list[c].boundingBox.x += (int)shift_x;
			object_list[c].boundingBox.y += (int)shift_y;

			// if bbs exceed img boundary after shift, then stop iteration
			if (object_list[c].boundingBox.x < 0 || object_list[c].boundingBox.y < 0 || object_list[c].boundingBox.br().x >= img.cols || object_list[c].boundingBox.br().y >= img.rows)
			{
				object_list[c].boundingBox &= Rect(0, 0, img.cols, img.rows); // make bbs be inside img

				//break;

				if (object_list[c].boundingBox.width < minObjWidth || object_list[c].boundingBox.height < minObjHeight)
				{
					delBbsOutImg = true;
					break; // if the part of bbs inside img is too small after shift, stop shift 	
				}

				if ((object_list[c].boundingBox.height & 1) == 0)    object_list[c].boundingBox.height -= 1; // bbs.height should be odd number
				if ((object_list[c].boundingBox.width & 1) == 0)    object_list[c].boundingBox.width -= 1; // bbs.width should be odd number

				CandCen = Point((object_list[c].boundingBox.width - 1) / 2, (object_list[c].boundingBox.height - 1) / 2);

				// if bbs has been changed, kernel should also be changed
				kernel.create(object_list[c].boundingBox.height, object_list[c].boundingBox.width, CV_64FC1);
				getKernel(kernel, kernel_type);
				// if bbs has been changed, weight should also be changed
				weight.create(object_list[c].boundingBox.height, object_list[c].boundingBox.width, CV_64FC1);
			}

			// if too small bbs center shift, then stop iteration
			if (pow(shift_x, 2) + pow(shift_y, 2) < epsilon)
			{
				//cout << "iter " << Mean_Shift_Iter << "   similarity" << similarity << endl;
				break;
			}

			// iterate at most Max_Mean_Shift_Iter times
			if (Mean_Shift_Iter == Max_Mean_Shift_Iter)
			{
				//cout << "iter " << Mean_Shift_Iter << "   similarity" << similarity << endl;
				break;
			}

			// compute color hist
			tempMat = img(object_list[c].boundingBox);
			computeHist(tempMat, kernel, hist[0]);

			// set weight
			setWeight(tempMat, kernel, object_list[c].hist, hist[0], weight);
		} //end of while

		if (delBbsOutImg)
		{
			for (int iterColor = 0; iterColor < 10; iterColor++)
			{
				if (objNumArray_BS[c] == objNumArray[iterColor])
				{
					objNumArray[iterColor] = 1000;
					break;
				}
			}
			object_list.erase(object_list.begin() + c);
			--c;
			continue; // if the part of bbs inside img is too small after scale and shift, abandon tracking this obj
		}

		// compute color hist
		tempMat = img(object_list[c].boundingBox);
		computeHist(tempMat, kernel, hist[0]);

		// choose scale with largest similarity to target model
		similarity = 0;
		for (int histIdx = 0; histIdx < histSize; ++histIdx) // compute similarity
		{
			similarity += sqrt(object_list[c].hist[histIdx] * hist[0][histIdx]);
		}
	} // for all obj

	return 1;
}

void MeanShiftTracker::getKernel(Mat &kernel, const int func_type)
{
	int H = kernel.rows - 1; // kernel.rows is odd 
	int W = kernel.cols - 1; // kernel.cols is odd

	switch (func_type)
	{
	case 0:
	{
		// Gaussian:  
		// sigma = x/3 as a gaussian is almost equal to 0 from 3 * sigma.
		double sig_w = (W / 2) / 3.0f;
		double sig_h = (H / 2) / 3.0f;
		double dev_w = sig_w*sig_w;
		double dev_h = sig_h*sig_h;

		for (int i = 0; i < kernel.rows; i++)
		{
			double yy = (i - H / 2)*(i - H / 2);

			for (int j = 0; j < kernel.cols; j++)
			{
				double xx = (j - W / 2)*(j - W / 2);
				kernel.at<double>(i, j) = exp(-.5f*(yy / dev_h + xx / dev_w));
			}
		}
		break;// Gaussian:  
	}
	case 1:
	{
		// Uniform:
		for (int i = 0; i < kernel.rows; i++)
		{
			for (int j = 0; j < kernel.cols; j++)
			{
				double HH = ((double(2 * i) / (double)H - 1.0f))*((double(2 * i) / (double)H - 1.0f));
				double WW = ((double(2 * j) / (double)W - 1.0f))*((double(2 * j) / (double)W - 1.0f));

				if (HH + WW <= 1)
					kernel.at<double>(i, j) = 1;
				else
					kernel.at<double>(i, j) = 0;
			}
		}
		break;
	}
	case 2:
	{
		// Epanechnikov:
		int w = W / 2, h = H / 2; // x-radius and y-radius  

		for (int i = 0; i < kernel.rows; i++)
		{
			for (int j = 0; j < kernel.cols; j++)
			{
				// scale to unit circle
				double dist_y = (double)(i - h) / h;
				double dist_x = (double)(j - w) / w;
				double distToCen = dist_x*dist_x + dist_y*dist_y; // distance from (i, j) to bbs center

				if (distToCen >= 1)    kernel.at<double>(i, j) = 0;
				else kernel.at<double>(i, j) = 2 * (1 - distToCen) / PI;
			}
		}
		break;
	}
	} // end of switch
}

void MeanShiftTracker::computeHist(const Mat &roiMat, const Mat &kernel, double hist[])
{
	if (roiMat.data == NULL) return;

	memset(hist, 0, histSize*sizeof(double)); // reset hist to 0
	double kernel_sum = 0; // sum for normalize

	if (roiMat.channels() == 3)
	{
		for (int i = 0; i < kernel.rows; i++)
		{
			for (int j = 0; j < kernel.cols; j++)
			{
				if (kernel.at<double>(i, j) == 0)	 continue;

				Vec3b bgr = roiMat.at<Vec3b>(i, j);
				int idx = (bgr.val[0] / bin_width)*bins*bins + (bgr.val[1] / bin_width)*bins + bgr.val[2] / bin_width;
				hist[idx] += kernel.at<double>(i, j);
				kernel_sum += kernel.at<double>(i, j);
			}
		}

		for (int i = 0; i < histSize; i++)
		{
			hist[i] /= kernel_sum;
		}
	}
	else // gray 
	{
		for (int i = 0; i < kernel.rows; i++)
		{
			for (int j = 0; j < kernel.cols; j++)
			{
				if (kernel.at<double>(i, j) == 0)	 continue;

				int idx = roiMat.at<uchar>(i, j) / bin_width;
				hist[idx] += kernel.at<double>(i, j);
				kernel_sum += kernel.at<double>(i, j);
			}
		}
		for (int i = 0; i < bins; i++)
		{
			hist[i] /= kernel_sum;
		}
	}
}

int MeanShiftTracker::setWeight(const Mat &roiMat, const Mat &kernel, const double tarHist[], const double candHist[], Mat &weight)
{
	if (roiMat.data == NULL) return -1;

	if (roiMat.channels() >= 3) // color
	{
		for (int i = 0; i < roiMat.rows; i++)
		{
			for (int j = 0; j < roiMat.cols; j++)
			{
				if (kernel.at<double>(i, j) == 0)	 continue;

				Vec3b bgr = roiMat.at<Vec3b>(i, j);
				int idx = (bgr.val[0] / bin_width)*bins*bins + (bgr.val[1] / bin_width)*bins + bgr.val[2] / bin_width;
				weight.at<double>(i, j) = sqrt(tarHist[idx] / candHist[idx]);
			}
		}
	}
	else // gray
	{
		for (int i = 0; i < roiMat.rows; i++)
		{
			for (int j = 0; j < roiMat.cols; j++)
			{
				if (kernel.at<double>(i, j) == 0)	 continue;

				int idx = roiMat.at<uchar>(i, j) / bin_width;
				weight.at<double>(i, j) = sqrt(tarHist[idx] / candHist[idx]);
			}
		}
	}

	return 0;
}

bool testBoxIntersection(int left1, int top1, int right1, int bottom1, int left2, int top2, int right2, int bottom2)
{
	if (right1 < left2)	return false;	// 1 is left of 2
	if (left1 > right2) return false;	// 1 is right of 2
	if (bottom1 < top2)	return false;	// 1 is above of 2
	if (top1 > bottom2) return false;	// 1 is below of 2
	return true;
}

bool MeanShiftTracker::testObjectIntersection(Object2D &obj1, Object2D &obj2)
{
	return testBoxIntersection(obj1.boundingBox.x, obj1.boundingBox.y, obj1.boundingBox.x + obj1.boundingBox.width - 1, obj1.boundingBox.y + obj1.boundingBox.height - 1,
		obj2.boundingBox.x, obj2.boundingBox.y, obj2.boundingBox.x + obj2.boundingBox.width - 1, obj2.boundingBox.y + obj2.boundingBox.height - 1);
}

bool MeanShiftTracker::testIntraObjectIntersection(vector<Object2D> &object_list, int cur_pos)
{
	bool bSection = false;
	for (size_t c = 0; c < object_list.size(); c++){
		if (c == cur_pos) continue;
		if (object_list[c].status == 3) continue; // avoid to delete two tracked objects
		bSection = testBoxIntersection(object_list[cur_pos].boundingBox.x, object_list[cur_pos].boundingBox.y, object_list[cur_pos].boundingBox.x + object_list[cur_pos].boundingBox.width - 1, object_list[cur_pos].boundingBox.y + object_list[cur_pos].boundingBox.height - 1,
			object_list[c].boundingBox.x, object_list[c].boundingBox.y, object_list[c].boundingBox.x + object_list[c].boundingBox.width - 1, object_list[c].boundingBox.y + object_list[c].boundingBox.height - 1);
		if (bSection)	break;
	}
	return bSection;
}

/* Function: overlayImage
*  Reference: http://jepsonsblog.blogspot.tw/2012/10/overlay-transparent-image-in-opencv.html
*  This code is applied to merge two images of different channel, only works if:
- The background is in BGR colour space.
- The foreground is in BGRA colour space. */
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location)
{
	background.copyTo(output);


	// start at the row indicated by location, or at row 0 if location.y is negative.
	for (int y = MAX(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if (fY >= foreground.rows)
			break;

		// start at the column indicated by location, 

		// or at column 0 if location.x is negative.
		for (int x = MAX(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // because of the translation.

			// we are done with this row if the column is outside of the foreground image.
			if (fX >= foreground.cols)
				break;

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity =
				((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

				/ 255.;


			// and now combine the background and foreground pixel, using the opacity, 

			// but only if opacity > 0.
			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] =
					(uchar)(backgroundPx * (1. - opacity) + foregroundPx * opacity);
			}
		}
	}
}

int Overlap(Rect a, Rect b, double ration)
{
	Rect c = a.x + a.width >= b.x + b.width ? a : b;
	Rect d = a.x + a.width >= b.x + b.width ? b : a;

	int e = MIN(d.x + d.width - c.x, d.width);
	if (e <= 0)
		return 0;

	c = a.y + a.height >= b.y + b.height ? a : b;
	d = a.y + a.height >= b.y + b.height ? b : a;

	int f = MIN(d.y + d.height - c.y, d.height);
	if (f <= 0)
		return 0;

	int overlapArea = e*f;
	int area_a = a.width * a.height;
	int area_b = b.width * b.height;
	int minArea = (area_a <= area_b ? area_a : area_b);

	if ((double)overlapArea / (double)minArea > ration) return 1;
	return 0;
}


void BubbleSort(int* array, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 1; j < size - i; j++)
		{
			if (array[j] < array[j - 1])
			{
				int temp = array[j];
				array[j] = array[j - 1];
				array[j - 1] = temp;
			}
		}
	}
}

//Refer to https://github.com/Myzhar/simple-opencv-kalman-tracker 
void KalmanF::Init()
{
	for (int i = 0; i < 5; i++)
	{
		// Transition State Matrix A
		// Note: set dT at each processing step!
		// [ 1 0 dT 0  0 0 ]
		// [ 0 1 0  dT 0 0 ]
		// [ 0 0 1  0  0 0 ]
		// [ 0 0 0  1  0 0 ]
		// [ 0 0 0  0  1 0 ]
		// [ 0 0 0  0  0 1 ]
		setIdentity(kf[i].transitionMatrix);

		// Measure Matrix H
		// [ 1 0 0 0 0 0 ]
		// [ 0 1 0 0 0 0 ]
		// [ 0 0 0 0 1 0 ]
		// [ 0 0 0 0 0 1 ]
		kf[i].measurementMatrix = Mat::zeros(measSize, stateSize, type);
		kf[i].measurementMatrix.at<float>(0) = 1.0f;
		kf[i].measurementMatrix.at<float>(7) = 1.0f;
		kf[i].measurementMatrix.at<float>(16) = 1.0f;
		kf[i].measurementMatrix.at<float>(23) = 1.0f;

		// Process Noise Covariance Matrix Q
		// [ Ex   0   0     0     0    0  ]
		// [ 0    Ey  0     0     0    0  ]
		// [ 0    0   Ev_x  0     0    0  ]
		// [ 0    0   0     Ev_y  0    0  ]
		// [ 0    0   0     0     Ew   0  ]
		// [ 0    0   0     0     0    Eh ]
		//setIdentity(kf.processNoiseCov, Scalar(1e-2));
		kf[i].processNoiseCov.at<float>(0) = (float)(1e-2);
		kf[i].processNoiseCov.at<float>(7) = (float)(1e-2);
		kf[i].processNoiseCov.at<float>(14) = 5.0f;
		kf[i].processNoiseCov.at<float>(21) = 5.0f;
		kf[i].processNoiseCov.at<float>(28) = (float)(1e-2);
		kf[i].processNoiseCov.at<float>(35) = (float)(1e-2);

		// Measures Noise Covariance Matrix R
		setIdentity(kf[i].measurementNoiseCov, Scalar(1e-1));
		// <<<< Kalman Filter
	}
}

void KalmanF::Predict(vector<Object2D> &object_list, vector<cv::Rect> &ballsBox)
{
	ticks = (double)cv::getTickCount();
	dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
	//cout << "dT: " << dT << endl;

	if (found)
	{
		//for (int i = 0; i < object_list.size(); i++)
		for (int i = 0; i < 10; i++)
		{
			// >>>> Matrix A
			kf[i].transitionMatrix.at<float>(2) = (float)dT;
			kf[i].transitionMatrix.at<float>(9) = (float)dT;
			// <<<< Matrix A

			//cout << "dT:" << endl << dT << endl;
			state[i] = kf[i].predict();
			//cout << "State post:" << endl << state[i] << endl;

			predRect[i].width = (int)(state[i].at<float>(4));
			predRect[i].height = (int)(state[i].at<float>(5));
			predRect[i].x = (int)(state[i].at<float>(0) - predRect[i].width / 2);
			predRect[i].y = (int)(state[i].at<float>(1) - predRect[i].height / 2);

			center[i].x = (int)(state[i].at<float>(0));
			center[i].y = (int)(state[i].at<float>(1));
		}
	}
}

void KalmanF::Update(vector<Object2D> &object_list, vector<cv::Rect> &ballsBox, int Upate)
{
	for (unsigned int iter = 0; iter < object_list.size(); iter++)
		ballsBox.push_back(object_list[iter].boundingBox);

	if (object_list.size() == 0)
	{
		notFoundCount++;
		if (notFoundCount >= 100)
			found = false;
	}
	else
	{
		notFoundCount = 0;
		static int stopFrame = 0;

		if ((Upate == 1) && (stopFrame == 0))
		{
			for (unsigned int i = 0; i < object_list.size(); i++)
			{
				meas[i].at<float>(0) = (float)(ballsBox[i].x + ballsBox[i].width / 2);
				meas[i].at<float>(1) = (float)(ballsBox[i].y + ballsBox[i].height / 2);
				meas[i].at<float>(2) = (float)ballsBox[i].width;
				meas[i].at<float>(3) = (float)ballsBox[i].height;

				if (!found) // First detection!
				{
					// >>>> Initialization
					kf[i].errorCovPre.at<float>(0) = 1; // px
					kf[i].errorCovPre.at<float>(7) = 1; // px
					kf[i].errorCovPre.at<float>(14) = 1;
					kf[i].errorCovPre.at<float>(21) = 1;
					kf[i].errorCovPre.at<float>(28) = 1; // px
					kf[i].errorCovPre.at<float>(35) = 1; // px

					state[i].at<float>(0) = meas[i].at<float>(0);
					state[i].at<float>(1) = meas[i].at<float>(1);
					state[i].at<float>(2) = 0;
					state[i].at<float>(3) = 0;
					state[i].at<float>(4) = meas[i].at<float>(2);
					state[i].at<float>(5) = meas[i].at<float>(3);
					// <<<< Initialization

					found = true;
				}
				else
					kf[i].correct(meas[i]); // Kalman Correction
				//cout << "Measure matrix:" << endl << meas[i] << endl;
			}
		}
		else
		{
			stopFrame++;
			if (stopFrame == 50)
				stopFrame = 0;
		}
	}
}
void KalmanF::drawPredBox(Mat &img)
{
	int i = 0;
	for (i = 0; i < 10; i++)
	{
		if ((predRect[i].x != 0) && (predRect[i].y != 0) && display_kalmanRectangle == true)
		{
			cv::circle(img, center[i], 2, CV_RGB(255, 0, 0), -1); // central point of red rectangle
			cv::rectangle(img, predRect[i], CV_RGB(255, 0, 0), 2); //red rectangle --> predict
		}

		if (display_kalmanArrow == true)
		{
			static int plot_arrow[10];
			if (plot_arrow[i] == true)
			{
				drawArrow(img, Point(pred_x[i], pred_y[i]), Point(center[i].x, center[i].y));
				pred_x[i] = center[i].x;
				pred_y[i] = center[i].y;
				plot_arrow[i] = false;
			}
			if ((predRect[i].x != 0) && (predRect[i].y != 0))
				plot_arrow[i] = true;
		}
	}
}
void drawArrow(Mat img, CvPoint p, CvPoint q)
{
	double angle; angle = atan2((double)p.y - q.y, (double)p.x - q.x);           //bevel angle of pq line
	double hypotenuse = sqrt((p.y - q.y)*(p.y - q.y) + (p.x - q.x)*(p.x - q.x)); //length of pq line

	/*The length of the arrow becomes three times from the original length */
	//	q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
	//	q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

	//	if ((hypotenuse < 80.0f) && (hypotenuse > 5.0f)) // Prevent drawing line of error 
	{
		/* Plot mainline */
		line(img, p, q, Scalar(0, 255, 0), 3, 1, 0);

		/* Plot two short lines */
		p.x = (int)(q.x + 9 * cos(angle + PI / 4));
		p.y = (int)(q.y + 9 * sin(angle + PI / 4));
		line(img, p, q, Scalar(0, 255, 0), 3, 1, 0);
		p.x = (int)(q.x + 9 * cos(angle - PI / 4));
		p.y = (int)(q.y + 9 * sin(angle - PI / 4));
		line(img, p, q, Scalar(0, 255, 0), 3, 1, 0);
	}
}

