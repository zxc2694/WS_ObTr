﻿#define _CRT_SECURE_NO_WARNINGS
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
#include "ObjectTracking.h"
#include "motionDetection.h"

int objNumArray[10];
int objNumArray_BS[10];
Scalar *ColorPtr;
bool newObjFind = false;

/* Function: tracking_function
* @param: img_input  - Original image (RGB, 640 X 480)
* @param: img_output - Image output with plotting trajectory (RGB, 640 X 480) 
* @param: fgmaskIpl  - Original BS image (RGB, 320 X 240)
* @param: bbs        - Detected ROIs. Input is a compressed size (320 X 240)
* @param: ObjNum  - The number of ROIs
*/
void tracking_function(Mat &img_input, Mat &img_output, IplImage *fgmaskIpl, CvRect *bbs, CvPoint *centers, int ObjNum)
{
	static char runFirst = true;
	static double t = 0;
	static vector<ObjTrackInfo> object_list;
	static KalmanF KF;
	static MeanShiftTracker ms_tracker(img_input.cols, img_input.rows, minObjWidth_Ini_Scale, minObjHeight_Ini_Scale, stopTrackingObjWithTooSmallWidth_Scale, stopTrackingObjWithTooSmallHeight_Scale);
	static Mat TrackingLine(img_input.rows, img_input.cols, CV_8UC4);
	TrackingLine = Scalar::all(0);
	t = (double)cvGetTickCount();   // Get executing time 

	ms_tracker.fgmaskIpl = fgmaskIpl;

	if (runFirst)
	{	
		// Arrange object number to prevent accumulation
		ObjNumArr(objNumArray, objNumArray_BS);

		// Kalman Filter initialization
		KF.Init();
	}

	// Enlarge the size of bbs "imgCompressionScale" times
	revertBbsSize(img_input, bbs, centers, ObjNum);

	// Main tracking code using Mean-shift algorithm
	ms_tracker.track(img_input, object_list);

	// if a obj has a upper bbs and a lower bbs in previous frms due to broken background subtraction output, merge them
	// and add new bbs into object_list for obj appearing for the 1st time in order to track it
	mergeBbsAndGetNewObjBbs(img_input, ms_tracker, object_list, bbs, ObjNum);

	// Modify the size of the tracking boxes and delete useless boxes
	ms_tracker.modifyTrackBox(img_input, ms_tracker, object_list, bbs, ObjNum);

	// Arrange object number to prevent accumulation
	ObjNumArr(objNumArray, objNumArray_BS);

	// Draw all the track boxes and their numbers 
	ms_tracker.drawTrackBox(img_input, object_list);

	//Plotting trajectories
	drawTrajectory(img_input, TrackingLine, ms_tracker, object_list);

	// Prediction and update of Kalman Filter 	
	KFtrack(img_input, object_list, KF);

	// Tracking image output (merge 3-channel image and 4-channel trakcing lines)
	overlayImage(img_input, TrackingLine, img_output, cv::Point(0, 0));

	runFirst = false;
	t = (double)cvGetTickCount() - t;
	cout << "tracking time = " << t / ((double)cvGetTickFrequency() *1000.) << "ms" << endl;

}

void revertBbsSize(Mat &img_input, CvRect *bbs, CvPoint *centers, int &ObjNum)
{
	// Enlarge the size of bbs 2 times
	for (int iter = 0; iter < ObjNum; ++iter)
	{
		bbs[iter].x *= imgCompressionScale;
		bbs[iter].y *= imgCompressionScale;
		bbs[iter].width *= imgCompressionScale;
		bbs[iter].height *= imgCompressionScale;
		centers[iter].x *= imgCompressionScale;
		centers[iter].y *= imgCompressionScale;
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

// if a obj has a upper bbs and a lower bbs in previous frms due to broken background subtraction output, merge them
// and add new bbs into object_list for obj appearing for the 1st time in order to track it
void mergeBbsAndGetNewObjBbs(Mat img_input, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list, CvRect *bbs, int ObjNum)
{
	// if a obj has a upper bbs and a lower bbs in previous frms due to broken background subtraction output, merge them
	int bbs_iter;
	size_t obj_list_iter;
	for (bbs_iter = 0; bbs_iter < ObjNum; ++bbs_iter)
	{
		bool Overlapping = false, addToList = true;
		vector<int> replaceList;

		for (obj_list_iter = 0; obj_list_iter < object_list.size(); ++obj_list_iter)
		{
			//If the height of bbs is 1.3 times lagrer than the height of boundingBox, determine whether replace the boundingBox by the following judgement
			if ((bbs[bbs_iter].height > 1.3f*object_list[(int)obj_list_iter].boundingBox.height)) 
			//if ((bbs[bbs_iter].width*bbs[bbs_iter].height > 1.8f*object_list[(int)obj_list_iter].boundingBox.width*object_list[(int)obj_list_iter].boundingBox.height)) //If the size of bbs is 1.8 times lagrer than the size of boundingBox, determine whether replace the boundingBox by the following judgement
				// && (bbs[bbs_iter].width*bbs[bbs_iter].height < 4.0f*object_list[obj_list_iter].boundingBox.width*object_list[obj_list_iter].boundingBox.height)
			{
				if (Overlap(bbs[bbs_iter], object_list[(int)obj_list_iter].boundingBox, 0.5f)) // Overlap > 0.5 --> replace the boundingBox
				{
					replaceList.push_back((int)obj_list_iter);
				}
			}
			else
			{
				// In else case, if the size of overlap is large, don't add to object list. (no replace)
				if (Overlap(bbs[bbs_iter], object_list[(int)obj_list_iter].boundingBox, 0.3f))		addToList = false; 
			}
		} // end of 2nd for 

		int iter1 = 0, iter2 = 0;

		if ((int)replaceList.size() != 0)
		{
			for (int iter = 0; iter < object_list.size(); ++iter)
			{
				//if ((bbs[bbs_iter].width*bbs[bbs_iter].height <= 1.8f*object_list[iter].boundingBox.width*object_list[iter].boundingBox.height) // contrary to above "if" above 
				if ((bbs[bbs_iter].height <= 1.3f*object_list[iter].boundingBox.height) // contrary to above "if" above 
					&& Overlap(bbs[bbs_iter], object_list[iter].boundingBox, 0.5f))		replaceList.push_back(iter);
			}

			for (iter1 = 0; iter1 < (int)replaceList.size(); ++iter1)
			{
				for (iter2 = iter1 + 1; iter2 < (int)replaceList.size(); ++iter2)
				{
					/*cout << Pixel32S(ms_tracker.DistMat, MIN(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
					MAX(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No)) << endl;*/

					if (Pixel32S(ms_tracker.DistMat, MIN(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
						MAX(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No)) > MAX_DIS_BET_PARTS_OF_ONE_OBJ)
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

			ms_tracker.updateObjBbs(img_input, object_list, bbs[bbs_iter], replaceList[objWithLongestDuration]);

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

		// and add new bbs into object_list for obj appearing for the 1st time in order to track it
		if (!Overlapping && addToList) 
		{
			ms_tracker.addTrackedList(img_input, object_list, bbs[bbs_iter], 2); // No replace and add object list -> bbs convert boundingBox.
			newObjFind = true;
		}

		vector<int>().swap(replaceList);
	}  // end of 1st for 
}

void MeanShiftTracker::modifyTrackBox(Mat img_input, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list, CvRect *bbs, int ObjNum)
{
	/* shrink the size of the tracking box */
	int bbsNumber;
	for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
	{
		bbsNumber = 0;
		int bbsIdxToReplace;

		for (int i = 0; i < ObjNum; i++)
		{
			// Find how many bbs in the tracking box
			if (Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.5f))
			{
				bbsIdxToReplace = i;
				bbsNumber++;
			}
			if (bbsNumber > 1)	break;
		}
		// When the width or height of tracking box is 1.1 times larger than the width or height of bbs
		if (bbsNumber == 1)
		{
			if ((object_list[obj_list_iter].boundingBox.width > 1.25f * bbs[bbsIdxToReplace].width) || 
				(object_list[obj_list_iter].boundingBox.height > 1.25f * bbs[bbsIdxToReplace].height))
				ms_tracker.updateObjBbs(img_input, object_list, bbs[bbsIdxToReplace], obj_list_iter); // Reset the scale of the tracking box.
		}
	}

	/* Removing motionless tracking box */
	static bool mergeBOX = false;
	static size_t leftObjNo, leftObjNum;
	bool checkDel = false;
	int black = 0, times = 0;
	for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
		//for (size_t obj_list_iter = object_list.size() - 1; object_list.size() == 0; obj_list_iter --)
	{
		if ((object_list[obj_list_iter].bIsUpdateTrack == true) && (object_list[obj_list_iter].PtCount != 0)) // Prevent to delete new object
		{
			for (int i = 0; i < ObjNum; i++)
			{
				if (Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.2f))
					break;
				else
				{
					black++; // Count the accumulation of no overlapping object
				}
			}
			// Restarting count when count > DELE_RECT_FRAMENO number
			if (object_list[obj_list_iter].cPtNumber == DELE_RECT_FRAMENO)
				object_list[obj_list_iter].cPtNumber = 0;

			// findBbs[i] = 0 -> no object; findBbs[i] = 1 -> has object
			if (black == ObjNum)
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
			// The following condition is that bbs and tracking box are not overlap from one to 'DELE_RECT_FRAMENO' consecutive frames.
			if ((1 <= times) && ((times < DELE_RECT_FRAMENO)) && ((keepTrajectory == true)))
			{
				int cenPointX = object_list[obj_list_iter].boundingBox.x + 0.5*object_list[obj_list_iter].boundingBox.width;
				int cenPointY = object_list[obj_list_iter].boundingBox.y + 0.5*object_list[obj_list_iter].boundingBox.height;

				// Directly remove the object which is near edges
				if ((cenPointX < img_input.cols * 0.1) || (cenPointX > img_input.cols * 0.9) || (cenPointY < img_input.rows * 0.1) || (cenPointY > img_input.rows * 0.9)) // Tracking box is on image edges
				{
					object_list_erase(object_list, obj_list_iter);
					checkDel = true;
				}
				// Tracking box is not on image edges
				else // merge tracking box from previous similar one
				{
					mergeBOX = true; // Wait for next object appearing
					leftObjNo = object_list[obj_list_iter].No; // Get the left object number
				}
			}

			// The following condition is that bbs and tracking box are not overlap for XX consecutive frames. (default: XX = 4).
			if ((DELE_RECT_FRAMENO == times) && (checkDel == false)) // no object for the long time 
			{
				object_list_erase(object_list, obj_list_iter); // Forcibly remove the object no matter what to do. 
				checkDel = true;
			}
			black = 0;
			times = 0;
		}
	}

	/* merge tracking box from previous similar one */
	bool bIsOverlap = false;
	if ((mergeBOX == true) && (newObjFind == true) && (checkDel == false))
	{
		// Due to the fact that the array number of object_list always updates in adding new object, we need to ensure what is the correct left object.
		for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
		{
			if (leftObjNo == object_list[obj_list_iter].No) // Find left object number
				leftObjNum = obj_list_iter;                 // Get real left object number called leftObjNum
		}
		for (int i = 0; i < ObjNum; i++)
		{
			if ((object_list.size() >= 2) && (Overlap(object_list[leftObjNum].boundingBox, bbs[i], 0.5f)))
			{
				bIsOverlap = true;
			}
		}
		if ((object_list.size() >= 2) && (bIsOverlap == false) && (object_list[leftObjNum].bIsUpdateTrack == true))
		{
			size_t newObjNum = (int)object_list.size() - 1;
			bool Similar = false;
			double similarityH = 0.0;

			for (int histIdx = 0; histIdx < MaxHistBins; ++histIdx) // Compute similarity
			{
				similarityH += sqrt(object_list[leftObjNum].hist[histIdx] * object_list[newObjNum].hist[histIdx]);
			}
			if (similarityH > 0.7)
			{
				ms_tracker.updateObjBbs(img_input, object_list, object_list[newObjNum].boundingBox, leftObjNum); //Reset the scale of the tracking box.
				object_list_erase(object_list, newObjNum);
				Similar = true;
			}
			//cout << object_list[leftObjNum].No << " to " << object_list[newObjNum].No << " similarity: " << similarityH << endl;
			similarityH = 0.0;

			if (!Similar)
				object_list_erase(object_list, leftObjNum);
		}
		mergeBOX = false;
	}

	newObjFind = false;
}

void drawTrajectory(Mat img_input, Mat &TrackingLine, MeanShiftTracker &ms_tracker, vector<ObjTrackInfo> &object_list)
{
	for (size_t obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
	{
		ms_tracker.drawTrackTrajectory(TrackingLine, object_list, obj_list_iter);

		// Position of plotting point
		int currentX = (int)(0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x));
		int currentY = (int)(0.4 * object_list[obj_list_iter].boundingBox.height + (object_list[obj_list_iter].boundingBox.y));

		/* Solve politting pulse of track */
		if (object_list[obj_list_iter].PtCount < 5)
		{
			// Get previous point in order to use line function. 
			object_list[obj_list_iter].pre_data_X = currentX;
			object_list[obj_list_iter].pre_data_Y = currentY;
		}
		else // Start judgement of plotting pulse after passing 5 frame
		{	
			// It decides whether the point Y is great change or not
			if ((30 >= abs(currentY - object_list[obj_list_iter].pre_data_Y)) || (30 >= abs(currentX - object_list[obj_list_iter].pre_data_X))) // Without great change -> Normal update point
			{
				object_list[obj_list_iter].pre_data_X = currentX;
				object_list[obj_list_iter].pre_data_Y = currentY;
			}
			else // Great change -> Not update point
			{
				object_list[obj_list_iter].waitFrame++; // Wait for plotting line
			}

			// Restart plotting suspended line
			if (object_list[obj_list_iter].waitFrame == 3)
			{
				object_list[obj_list_iter].pre_data_X = currentX;
				object_list[obj_list_iter].pre_data_Y = currentY;
				object_list[obj_list_iter].waitFrame = 0;
			}
		}

		// Restarting count when count > plotLineLength number
		if (object_list[obj_list_iter].PtNumber == plotLineLength + 1)
			object_list[obj_list_iter].PtNumber = 0;

		/* Run Bezier curve algorithm for getting smooth trajectory */
		Point smoothPoint[20];
		int currentPointer = object_list[obj_list_iter].PtNumber;
		if ((currentPointer > 10) && (currentPointer <= plotLineLength)) // the number of point array is from 10 ~ 99
		{
			// Use four points as input and obtain ten points as output for plotting
			Point p0 = object_list[obj_list_iter].point[currentPointer - 10];  //  1  2  3  4  5  6  7  8  9  10
			Point p1 = object_list[obj_list_iter].point[currentPointer - 7];   // use -  - use -  - use -  -  use
			Point p2 = object_list[obj_list_iter].point[currentPointer - 4];
			Point p3 = object_list[obj_list_iter].point[currentPointer - 1];

			// Normal case (assume max size =99)
			//  PtN   |     p3  p2  p1  p0
			//----------------------------
			// PtN=11:| use 10, 13, 16, 19
			// PtN=12:| use 11, 14, 17, 20
			// PtN=13:| use 12, 15, 18, 21
			// PtN=14:| use 13, 16, 19, 22
			// PtN=15:| use 14, 17, 20, 23
			// PtN=16:| use 15, 18, 21, 24
			// PtN=17:| use 16, 19, 22, 25
			// PtN=18:| use 17, 20, 23, 26
			// PtN=19:| use 18, 21, 24, 27
			// PtN=20:| use 19, 22, 25, 28
			// PtN=21:| use 20, 23, 26, 29

			BezierCurve(p0, p1, p2, p3, smoothPoint);

			for (int i = 0; i < 10; i++)
			{
				object_list[obj_list_iter].point[currentPointer - 10 + i] = smoothPoint[i];
			}
		}
		// the number of point array is from 0 ~ 10 after restarting counting
		else if ((object_list[obj_list_iter].PtCount > 10) && (currentPointer <= 10))
		{
			// Special case (assume max size =99)
			//  PtN   |     p3  p2  p1  p0
			//----------------------------
			// PtN =0:| use 99, 96, 93, 90
			// PtN =1:| use  0, 97, 94, 91
			// PtN =2:| use  1, 98, 95, 92
			// PtN =3:| use  2, 99, 96, 93
			// PtN =4:| use  3,  0, 97, 94
			// PtN =5:| use  4,  1, 98, 95
			// PtN =6:| use  5,  2, 99, 96
			// PtN =7:| use  6,  3,  0, 97
			// PtN =8:| use  7,  4,  1, 98
			// PtN =9:| use  8,  5,  2, 99
			// PtN=10:| use  9,  6,  3,  0

			Point p0 = currentPointer <= 9 ? object_list[obj_list_iter].point[plotLineLength - 9 + currentPointer] : object_list[obj_list_iter].point[currentPointer - 10];
			Point p1 = currentPointer <= 6 ? object_list[obj_list_iter].point[plotLineLength - 6 + currentPointer] : object_list[obj_list_iter].point[currentPointer - 7];
			Point p2 = currentPointer <= 3 ? object_list[obj_list_iter].point[plotLineLength - 3 + currentPointer] : object_list[obj_list_iter].point[currentPointer - 4];
			Point p3 = currentPointer == 0 ? object_list[obj_list_iter].point[plotLineLength + currentPointer] : object_list[obj_list_iter].point[currentPointer - 1];

			BezierCurve(p0, p1, p2, p3, smoothPoint);

			for (int i = 0; i < 10; i++)
			{
				//smooth[i]   i= 0  1  2  3  4  5  6  7  8  9 
				//PtN = 0 -->   90 91 92 93 94 95 96 97 98 99
				//PtN = 1 -->   91 92 93 94 95 96 97 98 99  0 
				//PtN = 2 -->   92 93 94 95 96 97 98 99  0  1
				//PtN = 3 -->   93 94 95 96 97 98 98  0  1  2
				//......                     ......
				//PtN = 8 -->   98 99  0  1  2  3  4  5  6  7 
				//PtN = 9 -->   99  0  1  2  3  4  5  6  7  8
				//PtN =10 -->    0  1  2  3  4  5  6  7  8  9
				int pointArrNum = i >= 10 - currentPointer ? i - (10 - currentPointer) : plotLineLength + currentPointer - 9 + i;
				object_list[obj_list_iter].point[pointArrNum] = smoothPoint[i];
			}
		}

		//Storage all of points on the array
		object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber] = Point(object_list[obj_list_iter].pre_data_X, object_list[obj_list_iter].pre_data_Y);
		object_list[obj_list_iter].PtNumber++;
		object_list[obj_list_iter].PtCount++;

	}// end of plotting trajectory

}

void KFtrack(Mat &img_input, vector<ObjTrackInfo> &object_list, KalmanF &KF)
{
	vector<cv::Rect> KFBox;

	/*Kalman Filter Function */
	if (use_Kalman == true)
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
				double hypotenuse = sqrt((double)((p_y - q_y)*(p_y - q_y) + (p_x - q_x)*(p_x - q_x))); //length of pq line
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
	histSize = bins*bins*bins + 1;

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

	/*ColorMatrix[0] = Scalar(0, 0, 255);
	ColorMatrix[1] = Scalar(255, 0, 0);
	ColorMatrix[2] = Scalar(0, 0, 255);
	ColorMatrix[3] = Scalar(255, 0, 0);
	ColorMatrix[4] = Scalar(0, 0, 255);
	ColorMatrix[5] = Scalar(255, 0, 0);
	ColorMatrix[6] = Scalar(0, 0, 255);
	ColorMatrix[7] = Scalar(255, 0, 0);
	ColorMatrix[8] = Scalar(0, 0, 255);
	ColorMatrix[9] = Scalar(255, 0, 0);*/

	ColorPtr = &ColorMatrix[0];

	Max_Mean_Shift_Iter = 8;
	Similar_Val_Threshold = 0.165;
	scaleBetFrame = 0.1f;
	scaleLearningRate = 0.5; //0.1 // scale change rate
	epsilon = 1.0;
	count = 0;
}

MeanShiftTracker::~MeanShiftTracker()
{
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

void MeanShiftTracker::addTrackedList(const Mat &img, vector<ObjTrackInfo> &object_list, Rect bbs, short type)
{
	// don't tracking too small obj 
	if (bbs.width < minObjWidth_Ini || bbs.height < minObjHeight_Ini)	return;

	// don't track when obj just emerge at img edge
	if (bbs.x < 3 || bbs.y < 3 || bbs.x + bbs.width > img.cols - 1 || bbs.y + bbs.height > img.rows - 1)		return;
	++count;

	if ((bbs.height & 1) == 0)    bbs.height -= 1; // bbs.height should be odd number
	if ((bbs.width & 1) == 0)    bbs.width -= 1; // bbs.width should be odd number

	ObjTrackInfo obj;
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
	obj.waitFrame = 0;
	obj.initFrame = 0;
	obj.kernel.create(obj.boundingBox.height, obj.boundingBox.width, CV_64FC1);
	obj.bIsUpdateTrack = true;
	obj.startOcc = true;

	for (int i = 0; i < DELE_RECT_FRAMENO; i++)
		obj.findBbs[i] = 1; // 1: has object; 0: no object -> default: has object

	getKernel(obj.kernel, kernel_type);

	Mat tempMat = img(obj.boundingBox);
	computeHist(tempMat, obj.boundingBox, obj.kernel, obj.hist);

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

	for (int iter = 0; iter < 10; iter++)
		objNumArray_BS[iter] = objNumArray[iter];

	BubbleSort(objNumArray_BS, 10);
}

void MeanShiftTracker::updateObjBbs(const Mat &img, vector<ObjTrackInfo> &object_list, Rect bbs, int idx)
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
	computeHist(tempMat, object_list[idx].boundingBox, object_list[idx].kernel, object_list[idx].hist);
}

void MeanShiftTracker::drawTrackBox(Mat &img, vector<ObjTrackInfo> &object_list)
{
	int iter;
	for (size_t c = 0; c < object_list.size(); c++)
	{
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
			stringstream ss;
			if (countNum == false)
			{
				for (iter = 0; iter < 10; iter++)
				{
					if (objNumArray_BS[c] == objNumArray[iter])
					{
						ss << iter + 1;
						break;
					}
				}
				object_list[c].color = *(ColorPtr + iter);
				rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
			}
			else // for debug
			{
				ss << object_list[c].No;
                rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
				putText(img, ss.str(), Point(object_list[c].boundingBox.x + object_list[c].boundingBox.width / 2 - 10, object_list[c].boundingBox.y + object_list[c].boundingBox.height / 2), 1, 3, object_list[c].color, 3);
			}
		}
	}
}

void MeanShiftTracker::drawTrackTrajectory(Mat &TrackingLine, vector<ObjTrackInfo> &object_list, size_t &obj_list_iter)
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

int MeanShiftTracker::track(Mat &img, vector<ObjTrackInfo> &object_list)
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
		double bestScale, similarity, largestSimilarity = 0.0; // choose best scale as the one with largest similarity to target model 
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
				int halfWidth = (object_list[c].initialBbsWidth - 1) / 2 * scale;
				int halfHeight = (object_list[c].initialBbsHeight - 1) / 2 * scale;
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
			computeHist(tempMat, CandBbs[scaleIter], kernel, hist[scaleIter]);

			// set weight
			weight.create(CandBbs[scaleIter].height, CandBbs[scaleIter].width, CV_64FC1);
			setWeight(tempMat, CandBbs[scaleIter], kernel, object_list[c].hist, hist[scaleIter], weight);

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

				double shift_x = normalizedShiftVec.x * CandCen.x; // denormalize bbs shift in img x-axis
				double shift_y = normalizedShiftVec.y * CandCen.y; // denormalize bbs shift in img y-axis

				//shift_x = round(shift_x);
				//shift_y = round(shift_y);

				if (shift_x >= 0)	shift_x = ceil(shift_x);
				else	shift_x = floor(shift_x);
				if (shift_y >= 0)	shift_y = ceil(shift_y);
				else	shift_y = floor(shift_y);

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
				computeHist(tempMat, CandBbs[scaleIter], kernel, hist[scaleIter]);

				// set weight
				setWeight(tempMat, CandBbs[scaleIter], kernel, object_list[c].hist, hist[scaleIter], weight);
			} //end of while

			if (delBbsOutImg)   continue; // if the part of bbs inside img is too small after scale and shift, abandon this scale and choose other scale

			// compute color hist
			tempMat = img(CandBbs[scaleIter]);
			computeHist(tempMat, CandBbs[scaleIter], kernel, hist[scaleIter]);

			// choose scale with largest similarity to target model
			similarity = 0.0;
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

		// if the part of bbs inside img is too small for all scales after shifts, stop tracking this obj, i.e. delete this obj from object_list 
		// if similarity < "ACCEPTABLE_SIMILARITY", stop tracking this obj, i.e. delete this obj from object_list 
		if (exceedImgBoundary || largestSimilarity < ACCEPTABLE_SIMILARITY)
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
			continue;
		}


		// determine scale by bestScale and scaleLearningRate
		object_list[c].objScale = scaleLearningRate*bestScale + (1 - scaleLearningRate)*object_list[c].objScale;


		// adopt candidate bbs scale determined above and implement Mean-Shift again
		int bbsCen_x = object_list[c].boundingBox.x + (object_list[c].boundingBox.width - 1) / 2;
		int bbsCen_y = object_list[c].boundingBox.y + (object_list[c].boundingBox.height - 1) / 2;
		int halfWidth = (object_list[c].initialBbsWidth - 1) / 2 * object_list[c].objScale;
		int halfHeight = (object_list[c].initialBbsHeight - 1) / 2 * object_list[c].objScale;
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
		computeHist(tempMat, object_list[c].boundingBox, kernel, hist[0]);

		// set weight
		weight.create(object_list[c].boundingBox.height, object_list[c].boundingBox.width, CV_64FC1);
		setWeight(tempMat, object_list[c].boundingBox, kernel, object_list[c].hist, hist[0], weight);

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

			double shift_x = normalizedShiftVec.x * CandCen.x; // denormalize shift in img x-axis
			double shift_y = normalizedShiftVec.y * CandCen.y; // denormalize shift in img y-axis

			//shift_x = round(shift_x);
			//shift_y = round(shift_y);

			if (shift_x >= 0)	shift_x = ceil(shift_x);
			else	shift_x = floor(shift_x);
			if (shift_y >= 0)	shift_y = ceil(shift_y);
			else	shift_y = floor(shift_y);

			//cout << "Mean_Shift_Iter " << shift_x << " " << shift_y << " ";

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

			//cout << "org boundingBox " << object_list[c].boundingBox.x << " " << object_list[c].boundingBox.y;


			object_list[c].boundingBox.x += (int)shift_x;
			object_list[c].boundingBox.y += (int)shift_y;

			//cout << " to " << object_list[c].boundingBox.x << " " << object_list[c].boundingBox.y << endl;


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
			if (pow(shift_x, 2) + pow(shift_y, 2) < epsilon)	break;

			// iterate at most Max_Mean_Shift_Iter times
			if (Mean_Shift_Iter == Max_Mean_Shift_Iter)		break;


			// compute color hist
			tempMat = img(object_list[c].boundingBox);
			computeHist(tempMat, object_list[c].boundingBox, kernel, hist[0]);

			// set weight
			setWeight(tempMat, object_list[c].boundingBox, kernel, object_list[c].hist, hist[0], weight);
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
		computeHist(tempMat, object_list[c].boundingBox, kernel, hist[0]);

		// compute resulting similarity
		similarity = 0.0;
		for (int histIdx = 0; histIdx < histSize; ++histIdx) // compute similarity
		{
			similarity += sqrt(object_list[c].hist[histIdx] * hist[0][histIdx]);
		}
		//cout << "final similarity " << similarity << endl;
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

void MeanShiftTracker::computeHist(const Mat &roiMat, const Rect &objBbs, const Mat &kernel, double hist[])
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
				if (fgmaskIpl->imageData[(objBbs.y / imgCompressionScale + i / imgCompressionScale)*fgmaskIpl->widthStep + (objBbs.x / imgCompressionScale + j / imgCompressionScale)] == 0)
				{
					hist[MaxHistBins - 1] += kernel.at<double>(i, j);
					kernel_sum += kernel.at<double>(i, j);
					continue;
				}

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

int MeanShiftTracker::setWeight(const Mat &roiMat, const Rect &objBbs, const Mat &kernel, const double tarHist[], const double candHist[], Mat &weight)
{
	if (roiMat.data == NULL) return -1;

	if (roiMat.channels() >= 3) // color
	{
		for (int i = 0; i < roiMat.rows; i++)
		{
			for (int j = 0; j < roiMat.cols; j++)
			{
				if (kernel.at<double>(i, j) == 0)	 continue;

				if (fgmaskIpl->imageData[(objBbs.y / 2 + i / 2)*fgmaskIpl->widthStep + (objBbs.x / 2 + j / 2)] == 0)
				{
					weight.at<double>(i, j) = sqrt(tarHist[MaxHistBins - 1] / candHist[MaxHistBins - 1]);
					continue;
				}

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

bool MeanShiftTracker::testObjectIntersection(ObjTrackInfo &obj1, ObjTrackInfo &obj2)
{
	return testBoxIntersection(obj1.boundingBox.x, obj1.boundingBox.y, obj1.boundingBox.x + obj1.boundingBox.width - 1, obj1.boundingBox.y + obj1.boundingBox.height - 1,
		obj2.boundingBox.x, obj2.boundingBox.y, obj2.boundingBox.x + obj2.boundingBox.width - 1, obj2.boundingBox.y + obj2.boundingBox.height - 1);
}

bool MeanShiftTracker::testIntraObjectIntersection(vector<ObjTrackInfo> &object_list, int cur_pos)
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

double OverlapValue(Rect a, Rect b)
{
	double ration;

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

	return (double)overlapArea / (double)minArea;
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

void KalmanF::Predict(vector<ObjTrackInfo> &object_list, vector<cv::Rect> &ballsBox)
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

void KalmanF::Update(vector<ObjTrackInfo> &object_list, vector<cv::Rect> &ballsBox, int Upate)
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
		if ((predRect[i].x != 0) && (predRect[i].y != 0) && (use_Kalman == true))
		{
			cv::circle(img, center[i], 2, CV_RGB(255, 0, 0), -1); // central point of red rectangle
			cv::rectangle(img, predRect[i], CV_RGB(255, 0, 0), 2); //red rectangle --> predict
		}

		if (use_Kalman == true)
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
	double hypotenuse = sqrt((double)((p.y - q.y)*(p.y - q.y) + (p.x - q.x)*(p.x - q.x))); //length of pq line

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

void object_list_erase(vector<ObjTrackInfo> &object_list, size_t &obj_list_iter)
{
	for (int iterColor = 0; iterColor < 10; iterColor++)
	{
		if (objNumArray_BS[obj_list_iter] == objNumArray[iterColor])
		{
			objNumArray[iterColor] = 1000; // Recover the value of which the number will be remove  
			break;
		}
	}
	object_list.erase(object_list.begin() + obj_list_iter);
}

// Plot smooth trajectory by Bézier curve algorithm 
void BezierCurve(Point p0, Point p1, Point p2, Point p3, Point *pointArr_output)
{
	int n = 0;
	for (float t = 0.0; t <= 1.0; t += 0.11)
	{
		float x = (-t*t*t + 3 * t*t - 3 * t + 1) * p0.x
			+ (3 * t*t*t - 6 * t*t + 3 * t)      * p1.x
			+ (-3 * t*t*t + 3 * t*t)             * p2.x
			+ (t*t*t)                            * p3.x;
		float y = (-t*t*t + 3 * t*t - 3 * t + 1) * p0.y
			+ (3 * t*t*t - 6 * t*t + 3 * t)      * p1.y
			+ (-3 * t*t*t + 3 * t*t)             * p2.y
			+ (t*t*t)                            * p3.y;

		pointArr_output[n].x = x;
		pointArr_output[n].y = y;
		n = n + 1;
	}
}