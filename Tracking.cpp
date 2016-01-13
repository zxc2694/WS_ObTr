#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include "Tracking.h"
#include <iostream>
#include <iomanip> 
#include <windows.h>

int objNumArray[10];
int objNumArray_BS[10];
CvBGCodeBookModel* model = 0;
Scalar *ColorPtr;

/* Function: tracking_function
 * @param: img        - Image input(RGB)
 * @param: fgmask     - Image after processing of background subtraction
 * @param: ms_tracker - Builded track object list pointer
 * @param: nframes    - The number of executions 
 * @param: ROI        - Input all ROIs
 * @param: ObjNum     - The number of objects
 * Note: 
 * 1. Both ROI and ObjNum are NULL, the funciton will use fgmask to find all ROIs of objects.
 * 2. Both ROI and ObjNum are not NULL, the function will ignore fgmask image.
*/
void tracking_function(Mat &img, Mat &fgmask, IObjectTracker *ms_tracker, int &nframes, CvRect *ROI, int ObjNum)
{
	Mat show_img;
	CvRect bbs[10], bbsV2[10];
	CvPoint centers[10];
	char outFilePath[100];
	char outFilePath2[100];
	int c, n, iter, iter2, MaxObjNum;
	static vector<Object2D> object_list;
	static vector<Object2D> prev_object_list;
	static char prevData = false;
	static int pre_data_X[10] = { 0 }, pre_data_Y[10] = { 0 };  //for tracking line
	static Mat background_BBS(img.rows, img.cols, CV_8UC1);
	static Mat TrackingLine(img.rows, img.cols, CV_8UC4);       // Normal: cols = 640, rows = 480
	
	// >>>> Kalman Filter
	vector<vector<cv::Point> > balls;
	vector<cv::Rect> ballsBox;
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
	unsigned int type = CV_32F;
	static double ticks = 0;
	static bool found = false;
	static int notFoundCount = 0;
	static KalmanFilter kf[5] = { KalmanFilter(stateSize, measSize, contrSize, type), KalmanFilter(stateSize, measSize, contrSize, type), KalmanFilter(stateSize, measSize, contrSize, type), KalmanFilter(stateSize, measSize, contrSize, type), KalmanFilter(stateSize, measSize, contrSize, type) };
	static Mat state[5] = { Mat(stateSize, 1, type), Mat(stateSize, 1, type), Mat(stateSize, 1, type), Mat(stateSize, 1, type), Mat(stateSize, 1, type) };
	static Mat meas[5] = { Mat(measSize, 1, type), Mat(measSize, 1, type), Mat(measSize, 1, type), Mat(measSize, 1, type), Mat(measSize, 1, type) };
	double precTick = ticks;
	ticks = (double)cv::getTickCount();
	double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
	// <<<< Kalman Filter
	
	TrackingLine = Scalar::all(0);
	IplImage *fgmaskIpl = &IplImage(fgmask);

	sprintf(outFilePath, "video_output//%05d.png", nframes + 1);
	sprintf(outFilePath2, "video_output//m%05d.png", nframes + 1);
	//sprintf(outFilePath, "video3_output//%05d.png", nframes + 180);
	//sprintf(outFilePath2, "video3_output//m%05d.png", nframes + 180);

	if (nframes == 0)
	{
		for (unsigned int s = 0; s < 10; s++)
		{
			objNumArray[s] = 65535;                       // Set all values as max number for ordered arrangement
			objNumArray_BS[s] = 65535;
		}
		KF_init(kf);
	}
	else if (nframes < nframesToLearnBG)
	{

	}
	else if (nframes == nframesToLearnBG)
	{
		if (ObjNum == NULL)                                                      //If ObjNum is NULL, we need to find all ROIs.
		{
			/* find components ,and compute bbs information  */
			MaxObjNum = 10;                                                         // less than 10 objects  
			find_connected_components(fgmaskIpl, 1, 4, &MaxObjNum, bbs, centers);

			for (int iter = 0; iter < MaxObjNum; ++iter)
			{
				ms_tracker->addTrackedList(img, object_list, bbs[iter], 2);
			}
		}
		ms_tracker->track(img, object_list);
	}
	else
	{
		if (ObjNum == NULL)                                                      //If ObjNum is NULL, we need to find all ROIs.
		{
			/* find components ,and compute bbs information  */
			MaxObjNum = 10;                                                      // less than 10 objects  
			find_connected_components(fgmaskIpl, 1, 4, &MaxObjNum, bbs, centers);

			Mat(fgmaskIpl).copyTo(background_BBS);                               // Copy fgmaskIpl to background_BBS
			static Mat srcROI[10];                                               // for shadow rectangle

			/* Eliminating people's shadow method */
			for (iter = 0; iter < MaxObjNum; iter++){                            // Get all shadow rectangles named bbsV2
				bbsV2[iter].x = bbs[iter].x;
				bbsV2[iter].y = bbs[iter].y + bbs[iter].height * 0.75;
				bbsV2[iter].width = bbs[iter].width;
				bbsV2[iter].height = bbs[iter].height * 0.25;
				srcROI[iter] = background_BBS(Rect(bbsV2[iter].x, bbsV2[iter].y, bbsV2[iter].width, bbsV2[iter].height)); // srcROI is depended on the image of background_BBS
				srcROI[iter] = Scalar::all(0);                                  // Set srcROI as showing black color
			}
			IplImage *BBSIpl = &IplImage(background_BBS);
			find_connected_components(BBSIpl, 1, 4, &MaxObjNum, bbs, centers);  // Secondly, Run the function of searching components to get update of bbs

			for (iter = 0; iter < MaxObjNum; iter++)
				bbs[iter].height = bbs[iter].height + bbsV2[iter].height;       // Merge bbs and bbsV2 to get final ROI
          
			if (display_bbsRectangle == true)
			{
				/* Plot the rectangles background subtarction finds */
				for (iter = 0; iter < MaxObjNum; iter++){
					rectangle(img, bbs[iter], Scalar(0, 255, 255), 2);
				}
			}
		}
		else   //If ObjNum is not NULL, we use existing ROIs.
		{
			MaxObjNum = ObjNum;
			for (int iter = 0; iter < MaxObjNum; iter++)
				bbs[iter] = ROI[iter];
		}

		ms_tracker->track(img, object_list);

		if (nframes > nframesToLearnBG + 1) //Start to update the object tracking
			ms_tracker->checkTrackedList(object_list, prev_object_list);

		int bbs_iter;
		size_t obj_list_iter;
		for (bbs_iter = 0; bbs_iter < MaxObjNum; ++bbs_iter)
		{
			bool Overlapping = false, addToList = true;
			vector<int> replaceList;

			for (obj_list_iter = 0; obj_list_iter < object_list.size(); ++obj_list_iter)
			{
				if ((bbs[bbs_iter].width*bbs[bbs_iter].height > 1.8f*object_list[(int)obj_list_iter].boundingBox.width*object_list[(int)obj_list_iter].boundingBox.height)) //If the size of bbs is 1.8 times lagrer than the size of boundingBox, replace the boundingBox.
					// && (bbs[bbs_iter].width*bbs[bbs_iter].height < 4.0f*object_list[obj_list_iter].boundingBox.width*object_list[obj_list_iter].boundingBox.height)
				{
					if (Overlap(bbs[bbs_iter], object_list[(int)obj_list_iter].boundingBox, 0.5f)) // Overlap > 0.5 --> replace the boundingBox
					{
						replaceList.push_back((int)obj_list_iter);
					}
				}
				else
				{
					if (Overlap(bbs[bbs_iter], object_list[(int)obj_list_iter].boundingBox, 0.3f))		addToList = false; //If the size of overlap is small, don't add to object list. (no replace)
				}
			} // end of 2nd for 

			int iter1 = 0, iter2 = 0;

			if ((int)replaceList.size() != 0)
			{
				for (int iter = 0; iter < object_list.size(); ++iter)
				{
					if ((bbs[bbs_iter].width*bbs[bbs_iter].height <= 1.8f*object_list[iter].boundingBox.width*object_list[iter].boundingBox.height)
						&& Overlap(bbs[bbs_iter], object_list[iter].boundingBox, 0.5f))		replaceList.push_back(iter);
				}

				for (iter1 = 0; iter1 < (int)replaceList.size(); ++iter1)
				{
					for (iter2 = iter1 + 1; iter2 < (int)replaceList.size(); ++iter2)
					{
//						cout << Pixel32S(ms_tracker->DistMat, min(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
//							max(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No)) << endl;

						if (Pixel32S(ms_tracker->DistMat, min(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
							max(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No)) > MAX_DIS_BET_PARTS_OF_ONE_OBJ)
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
				ms_tracker->updateObjBbs(img, object_list, bbs[bbs_iter], replaceList[0]);
				for (int iter = 1; iter < (int)replaceList.size(); ++iter)
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

			if (!Overlapping && addToList)		ms_tracker->addTrackedList(img, object_list, bbs[bbs_iter], 2); //No replace and add object list -> bbs convert boundingBox.

			vector<int>().swap(replaceList);
		}  // end of 1st for 

		for (iter = 0; iter < 10; iter++)
			objNumArray_BS[iter] = objNumArray[iter]; // Copy array from objNumArray to objNumArray_BS

		BubbleSort(objNumArray_BS, 10);               // Let objNumArray_BS array execute bubble sort 

		ms_tracker->drawTrackBox(img, object_list);   // Draw all the track boxes and their numbers 


		/* plotting trajectory */
		for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
		{
			if (prevData == true) //prevent plotting tracking line when previous tracking data is none.
			{
				// Plotting all the tracking lines
				ms_tracker->drawTrackTrajectory(TrackingLine, object_list, obj_list_iter);
				
				// Removing the tracking box when it's motionless for a while 
				if ((object_list[obj_list_iter].comparePoint[0].x != 0) && (object_list[obj_list_iter].comparePoint[0] == object_list[obj_list_iter].comparePoint[DELE_RECT_FRAMENO]) && (object_list[obj_list_iter].comparePoint[DELE_RECT_FRAMENO/2] == object_list[obj_list_iter].comparePoint[DELE_RECT_FRAMENO]))
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
				}


			}
			if (object_list.size() == 0){ //Prevent out of vector range
				break;
			}
			// Get previous point in order to use line function. 
			pre_data_X[obj_list_iter] = 0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x);
			pre_data_Y[obj_list_iter] = 0.9 * object_list[obj_list_iter].boundingBox.height + (object_list[obj_list_iter].boundingBox.y);
			
			//Restarting count when count > plotLineLength number
			if (object_list[obj_list_iter].PtNumber == plotLineLength + 1) 
				object_list[obj_list_iter].PtNumber = 0;

			object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber] = Point(pre_data_X[obj_list_iter], pre_data_Y[obj_list_iter]); //Storage all of points on the array. 
			
			//Restarting count when count > DELE_RECT_FRAMENO number
			if (object_list[obj_list_iter].cPtNumber == DELE_RECT_FRAMENO + 1) 
				object_list[obj_list_iter].cPtNumber = 0;
			
			object_list[obj_list_iter].comparePoint[object_list[obj_list_iter].cPtNumber] = Point(pre_data_X[obj_list_iter], pre_data_Y[obj_list_iter]); //Storage all of points on the array. 

			object_list[obj_list_iter].PtNumber++;
			object_list[obj_list_iter].cPtNumber++;
			object_list[obj_list_iter].PtCount++;

		}// end of plotting trajectory
		prevData = true;

		// >>>>> Kalman Update
		// >>>> Kalman Filter
		//if (found)
		//{
		//	for (int i = 0; i < object_list.size(); i++)
		//	{
		//		// >>>> Matrix A
		//		kf[i].transitionMatrix.at<float>(2) = dT;
		//		kf[i].transitionMatrix.at<float>(9) = dT;
		//		// <<<< Matrix A

		//		//cout << "dT:" << endl << dT << endl;
		//		state[i] = kf[i].predict();
		//		//cout << "State post:" << endl << state[i] << endl;

		//		cv::Rect predRect;
		//		predRect.width = state[i].at<float>(4);
		//		predRect.height = state[i].at<float>(5);
		//		predRect.x = state[i].at<float>(0) - predRect.width / 2;
		//		predRect.y = state[i].at<float>(1) - predRect.height / 2;

		//		cv::Point center;
		//		center.x = state[i].at<float>(0);
		//		center.y = state[i].at<float>(1);
		//		if ((predRect.x != 0) && (predRect.y != 0) && display_kalmanRectangle == true)
		//		{
		//			cv::circle(img, center, 2, CV_RGB(255, 0, 0), -1); // central point of red rectangle
		//			cv::rectangle(img, predRect, CV_RGB(255, 0, 0), 2); //red rectangle --> predict
		//		}
		//	}
		//}
		//for (int iter = 0; iter < object_list.size(); iter++){
		//	ballsBox.push_back(object_list[iter].boundingBox);
		//}
		//// <<<< Kalman Filter
		//if (object_list.size() == 0)
		//{
		//	notFoundCount++;
		//	if (notFoundCount >= 100)
		//		found = false;
		//}
		//else
		//{
		//	notFoundCount = 0;
		//	for (int i = 0; i < object_list.size(); i++)
		//	{
		//		meas[i].at<float>(0) = ballsBox[i].x + ballsBox[i].width / 2;
		//		meas[i].at<float>(1) = ballsBox[i].y + ballsBox[i].height / 2;
		//		meas[i].at<float>(2) = (float)ballsBox[i].width;
		//		meas[i].at<float>(3) = (float)ballsBox[i].height;

		//		if (!found) // First detection!
		//		{
		//			// >>>> Initialization
		//			kf[i].errorCovPre.at<float>(0) = 1; // px
		//			kf[i].errorCovPre.at<float>(7) = 1; // px
		//			kf[i].errorCovPre.at<float>(14) = 1;
		//			kf[i].errorCovPre.at<float>(21) = 1;
		//			kf[i].errorCovPre.at<float>(28) = 1; // px
		//			kf[i].errorCovPre.at<float>(35) = 1; // px

		//			state[i].at<float>(0) = meas[i].at<float>(0);
		//			state[i].at<float>(1) = meas[i].at<float>(1);
		//			state[i].at<float>(2) = 0;
		//			state[i].at<float>(3) = 0;
		//			state[i].at<float>(4) = meas[i].at<float>(2);
		//			state[i].at<float>(5) = meas[i].at<float>(3);
		//			// <<<< Initialization

		//			found = true;
		//		}
		//		else
		//			kf[i].correct(meas[i]); // Kalman Correction
		//		//cout << "Measure matrix:" << endl << meas[i] << endl;
		//	}
		//}
		// <<<<< Kalman Update
	}

	/* Show the number of the frame on the image */
	stringstream textFrameNo;
	textFrameNo << nframes;
	putText(img, "Frame=" + textFrameNo.str(), Point(10, img.rows - 10), 1, 1, Scalar(0, 0, 255), 1); //Show the number of the frame on the picture

	/* Display image output */
	overlayImage(img, TrackingLine, show_img, cv::Point(0, 0)); // Merge 3-channel image and 4-channel image
	imshow("image", show_img);
	cvShowImage("foreground mask", fgmaskIpl);

	imwrite(outFilePath, show_img);
	cvSaveImage(outFilePath2, fgmaskIpl);

}

MeanShiftTracker::MeanShiftTracker() :kernel_type(0), radius(1), bin_width(32), count(0), Max_Iters(8), Similar_Val_Threshold(0.165)
{
	bins = 256 / bin_width;
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
}

int MeanShiftTracker::DistBetObj(Rect a, Rect b)
{
	int c, d;
	if ((a.x > b.x + b.width || b.x > a.x + a.width) || (a.y > b.y + b.height || b.y > a.y + a.height))
	{
		if (!(a.x > b.x + b.width || b.x > a.x + a.width))		c = 0;
		else	c = min(abs(a.x - (b.x + b.width)), abs((a.x + a.width) - b.x));

		if (!(a.y > b.y + b.height || b.y > a.y + a.height))	d = 0;
		else    d = min(abs(a.y - (b.y + b.height)), abs((a.y + a.height) - b.y));

		return sqrt((float)(c*c + d*d));
	}
	else return 0;
}

void MeanShiftTracker::addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type)
{
	++count;

	Object2D obj;
	obj.No = count;
	//CvScalar Scalar = cvScalar(rand() % 256, rand() % 256, rand() % 256);
	obj.color = ColorMatrix[count%10];
	obj.type = type;
	obj.status = 3;
	obj.boundingBox = bbs; 
	obj.xyz.z = 20;
	obj.PtNumber = 0;
	obj.cPtNumber = 0;
	obj.PtCount = 0;
	obj.times = 1;
	memset(obj.hist, 0, MaxHistBins*sizeof(int));

	Mat kernel(obj.boundingBox.height, obj.boundingBox.width, CV_32FC1);
	getParzenWindow(kernel, radius, kernel_type);
	Mat tempMat = img(obj.boundingBox);
	getDensityEstimate(tempMat, kernel, obj.hist);

    object_list.push_back(obj);	

	for (size_t iter = 0; iter < object_list.size(); ++iter)
	{ 
		if (obj.No < object_list[(int)iter].No)
			Pixel32S(DistMat, obj.No, object_list[(int)iter].No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);
		else if (obj.No > object_list[(int)iter].No) 
			Pixel32S(DistMat, object_list[(int)iter].No, obj.No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);

//		cout << Pixel32S(DistMat, object_list[(int)iter].No, obj.No);
//		cout << "";

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
	object_list[idx].boundingBox = bbs;
	memset(object_list[idx].hist, 0, MaxHistBins*sizeof(int));
	Mat kernel(object_list[idx].boundingBox.height, object_list[idx].boundingBox.width, CV_32FC1);
	getParzenWindow(kernel, radius, kernel_type);
	Mat tempMat = img(object_list[idx].boundingBox);
	getDensityEstimate(tempMat, kernel, object_list[idx].hist);
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
			if (object_list[c].similar_val > Similar_Val_Threshold-0.02 && !bIntraSec)
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

			if (object_list[c].type == 1){ //vehicle				
				std::stringstream ss,ss1,ss2,ss3;			
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
			if (object_list[c].type == 2){ //pedestrian
				std::stringstream ss, ss1, ss2, ss3;

				//ss << std::fixed << std::setprecision(2) << object_list[c].xyz.z;
				//cv::putText(img, "car:" + ss.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y - 8), 1, 1, ColorMatrix[c]); //object_list[c].color
				//ss1 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.x;
				//ss2 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.y;
				//cv::putText(img, "prob:" + ss1.str() + "," + ss2.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, ColorMatrix[c]);		
				for (iter = 0; iter < 10; iter++)
				{
					if (objNumArray_BS[c] == objNumArray[iter])
					{
						ss3 << iter + 1;
						break;
					}
				}
				object_list[c].color = *(ColorPtr + iter);

            //    ss3 << object_list[c].No;
				cv::rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
				cv::putText(img, ss3.str(), Point(object_list[c].boundingBox.x + object_list[c].boundingBox.width / 2 - 10, object_list[c].boundingBox.y + object_list[c].boundingBox.height / 2), 1, 3, object_list[c].color, 3);

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
	for (size_t c = 0; c < object_list.size(); c++)
	{
		//if (object_list[c].status == 3)
		//{ // miss detected
			Rect Pos = object_list[c].boundingBox;
			Mat kernel(Pos.height, Pos.width, CV_32FC1);
			getParzenWindow(kernel, radius, kernel_type);
			//
			//Mat grad_x, grad_y; // gradient of kernel
			//cv::Sobel(kernel, grad_x, CV_32F, 1, 0, 3);
			//cv::Sobel(kernel, grad_y, CV_32F, 0, 1, 3);
			//grad_x = -grad_x;
			//grad_y = -grad_y;
			//
			Mat kernerX = (Mat_<float>(1, 3) << -1, 0, 1);
			Mat kernerY = (Mat_<float>(3, 1) << -1, 0, 1);
			Mat grad_x, grad_y; // gradient of kernel
			cv::filter2D(kernel, grad_x, CV_32F, kernerX); // grad_x is kernel's gradient wrt x-axis
			cv::filter2D(kernel, grad_y, CV_32F, kernerY); // grad_y is kernel's gradient wrt y-axis
			grad_x = -grad_x;
			grad_y = -grad_y;

			Mat tempMat = img(Pos);
			float hist[MaxHistBins] = { 0 };
			getDensityEstimate(tempMat, kernel, hist);
			Mat weight(Pos.height, Pos.width, CV_32FC1);
			object_list[c].similar_val = computeSimilarity(tempMat, kernel, object_list[c].hist, hist, weight);
//			cout << object_list[c].similar_val << endl;
			int iters = 0;
			while (iters < Max_Iters && object_list[c].similar_val < Similar_Val_Threshold)
			{
				float num_x = 0;
				float num_y = 0;
				float den = 0;
				for (int i = 0; i < kernel.rows; i++){
					for (int j = 0; j < kernel.cols; j++){
						num_x += i*weight.at<float>(i, j)*grad_x.at<float>(i, j); // grad_x is kernel's gradient wrt x-axis
						num_y += j*weight.at<float>(i, j)*grad_y.at<float>(i, j); // grad_y is kernel's gradient wrt y-axis
						//double val = cv::norm(grad_x.at<float>(i, j), grad_y.at<float>(i, j));
						float val = sqrt((grad_x.at<float>(i, j) - grad_y.at<float>(i, j))*(grad_x.at<float>(i, j) - grad_y.at<float>(i, j)));
						den += weight.at<float>(i, j)*val;
					}
				}
				//
				den += 1E-05;
				float dx = num_x / den;
				float dy = num_y / den;
				Pos.x += cvRound(dx);
				Pos.y += cvRound(dy);
				if (Pos.x < 0 || Pos.x + Pos.width > img.cols - 1 || Pos.y < 0 || Pos.y + Pos.height>img.rows - 1){
					//object_list[c].status = 4; // track lossed
					break; // break while
				}
				tempMat = img(Pos);
				getDensityEstimate(tempMat, kernel, hist);
				object_list[c].similar_val = computeSimilarity(tempMat, kernel, object_list[c].hist, hist, weight);
				object_list[c].boundingBox = Pos;
				iters++;
//				cout << iters << ":" << object_list[c].similar_val << "...";
			} //end of while
//			cout << Pos << endl;
		//} // end of if
	} // end of for
	return 1;
}

void MeanShiftTracker::getParzenWindow(Mat &kernel, const int R, const int func_type)
{
	int H = kernel.rows;
	int W = kernel.cols;
	switch (func_type){
	case 0:
	{
		// Gaussian:  
		// sigma = x/3 as a gaussian is almost equal to 0 from 3 * sigma.
		float sig_w = (float(R*W) / 2.0f) / 3.0f;
		float sig_h = (float(R*H) / 2.0f) / 3.0f;
		float dev_w = sig_w*sig_w;
		float dev_h = sig_h*sig_h;
		for (int i = 0; i < H; i++){
			float yy = (i - .5f*H)*(i - .5f*H);
			for (int j = 0; j < W; j++){
				float xx = (j - .5f*W)*(j - .5f*W);
				kernel.at<float>(i, j) = exp(-.5f*(yy / dev_h + xx / dev_w));
			}
		}
		break;// Gaussian:  
	}
	case 1:
	{
		// Uniform:
		for (int i = 0; i < H; i++){
			for (int j = 0; j < W; j++){
				float HH = ((float(2 * i) / (float)H - 1.0f) / (float)R)*((float(2 * i) / (float)H - 1.0f) / (float)R);
				float WW = ((float(2 * j) / (float)W - 1.0f) / (float)R)*((float(2 * j) / (float)W - 1.0f) / (float)R);
				if (HH + WW <= 1)
					kernel.at<float>(i, j) = 1;
				else
					kernel.at<float>(i, j) = 0;
			}
		}
		break;
	}
	case 2:
	{
		// Epanechnikov:
		for (int i = 0; i < H; i++){
			for (int j = 0; j < W; j++){
				float RH2 = (float(2 * i) / float(R*H) - 1.0f / (float)R)*(float(2 * i) / float(R*H) - 1.0f / (float)R);
				float RW2 = (float(2 * j) / float(R*W) - 1.0f / (float)R)*(float(2 * j) / float(R*W) - 1.0f / (float)R);
				kernel.at<float>(i, j) = (1 - RH2 - RW2);
				if (kernel.at<float>(i, j) < 0)
					kernel.at<float>(i, j) = 0;
			}
		}
		break;
	}
	}
}

void MeanShiftTracker::getDensityEstimate(const Mat &roiMat, const Mat &kernel, float hist[])
{
	if (roiMat.data == NULL) return;
	float kernel_sum = 0;
	if (roiMat.channels() == 3){
		for (int i = 0; i < kernel.rows; i++){
			for (int j = 0; j < kernel.cols; j++){
				Vec3b bgr = roiMat.at<Vec3b>(i, j);
				int idx = (bgr.val[0] / bin_width)*bins*bins + (bgr.val[1] / bin_width)*bins + bgr.val[2] / bin_width;
				hist[idx] += kernel.at<float>(i, j);
				kernel_sum += kernel.at<float>(i, j);
			}
		}
		for (int i = 0; i < bins*bins*bins; i++){
			hist[i] /= kernel_sum;
		}
	}
	else{ // gray 
		for (int i = 0; i < kernel.rows; i++){
			for (int j = 0; j < kernel.cols; j++){
				int idx = roiMat.at<uchar>(i, j) / bin_width;
				hist[idx] += kernel.at<float>(i, j);
				kernel_sum += kernel.at<float>(i, j);
			}
		}
		for (int i = 0; i < bins; i++){
			hist[i] /= kernel_sum;
		}
	}
}

double MeanShiftTracker::computeSimilarity(const Mat &roiMat, const Mat &kernel, const float target[], const float candidate[], Mat &weight)
{
	if (roiMat.data == NULL) return -1;
	double similar_val = 0;
	if (roiMat.channels() >= 3){
		for (int i = 0; i < roiMat.rows; i++){
			for (int j = 0; j < roiMat.cols; j++){
				Vec3b bgr = roiMat.at<Vec3b>(i, j);
				int idx = (bgr.val[0] / bin_width)*bins*bins + (bgr.val[1] / bin_width)*bins + bgr.val[2] / bin_width;
				weight.at<float>(i, j) = sqrt((float)(target[idx]) / (float)(candidate[idx] + 1e-5));
				/*if ((float)candidate[idx] == 0)
				weight.at<float>(i, j) = sqrt((float)(target[idx] + 1e-5) / (float)(candidate[idx] + 1e-5));
				else
				weight.at<float>(i, j) = sqrt((float)target[idx] / (float)candidate[idx]);*/
				similar_val += weight.at<float>(i, j)*kernel.at<float>(i, j);
			}
		}
	}
	else{
		for (int i = 0; i < roiMat.rows; i++){
			for (int j = 0; j < roiMat.cols; j++){
				int idx = roiMat.at<uchar>(i, j) / bin_width;
				weight.at<float>(i, j) = sqrt((float)(target[idx] + 1) / (float)(candidate[idx] + 1));
				similar_val += weight.at<float>(i, j)*kernel.at<float>(i, j);
			}
		}
	}
	return similar_val / double(roiMat.rows*roiMat.cols);
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

void CodeBookInit()
{
	model = cvCreateBGCodeBookModel();
	//Set color thresholds to default values
	model->modMin[0] = 3;
	model->modMin[1] = model->modMin[2] = 3;
	model->modMax[0] = 10;
	model->modMax[1] = model->modMax[2] = 10;
	model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;
}

void RunCodeBook(IplImage* &image, IplImage* &yuvImage, IplImage* &ImaskCodeBook, IplImage* &ImaskCodeBookCC, int &nframes)
{
	if (nframes == 0)
	{
		// CODEBOOK METHOD ALLOCATION
		yuvImage = cvCloneImage(image);
		ImaskCodeBook = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
		ImaskCodeBookCC = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
		cvSet(ImaskCodeBook, cvScalar(255));
	}
	cvCvtColor(image, yuvImage, CV_BGR2YCrCb);//YUV For codebook method
	//This is where we build our background model
	if (nframes < nframesToLearnBG)
		cvBGCodeBookUpdate(model, yuvImage);

	if (nframes == nframesToLearnBG)
		cvBGCodeBookClearStale(model, model->t / 2);

	//Find the foreground if any
	if (nframes >= nframesToLearnBG)
	{
		// Find foreground by codebook method
		cvBGCodeBookDiff(model, yuvImage, ImaskCodeBook);
		// This part just to visualize bounding boxes and centers if desired
		cvCopy(ImaskCodeBook, ImaskCodeBookCC);
		cvSegmentFGMask(ImaskCodeBookCC);
	}
}

void find_connected_components(IplImage *mask, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers)
{
	static CvMemStorage* mem_storage = NULL;
	static CvSeq* contours = NULL;

	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_OPEN, CVCLOSE_ITR);    //clear up raw mask
	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_CLOSE, CVCLOSE_ITR);

	/* find contours around only bigger regions */
	if (mem_storage == NULL)
	{
		mem_storage = cvCreateMemStorage(0);
	}
	else
		cvClearMemStorage(mem_storage);

	CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	CvSeq* c;
	int numCont = 0;

	while ((c = cvFindNextContour(scanner)) != NULL)
	{
		double len = cvContourPerimeter(c);
		double q = (mask->height + mask->width) / perimScale; // calculate perimeter len threshold

		/* Get rid of blob if its perimeter is too small: */
		if (len < q)
			cvSubstituteContour(scanner, NULL);

		else
		{
			/* Smooth its edges if its large enough */
			CvSeq* c_new;
			if (poly1_hull0) {
				c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage, CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL, 0); // Polygonal approximation
			}
			else {
				c_new = cvConvexHull2(c, mem_storage, CV_CLOCKWISE, 1); // Convex Hull of the segmentation
			}
			cvSubstituteContour(scanner, c_new);
			numCont++;
		}
	}
	contours = cvEndFindContours(&scanner);
	const CvScalar CVX_WHITE = CV_RGB(0xff, 0xff, 0xff);
	const CvScalar CVX_BLACK = CV_RGB(0x00, 0x00, 0x00);

	/* Paint the found regions back into image */
	cvZero(mask);
	IplImage *maskTemp;

	/* Calc center of mass AND/OR bounding rectangles*/
	if (num != NULL) {
		int N = *num, numFilled = 0, i = 0;
		CvMoments moments;
		double M00, M01, M10;
		maskTemp = cvCloneImage(mask);
		for (i = 0, c = contours; c != NULL; c = c->h_next, i++) //User wants to collect statistics
		{
			if (i < N)
			{
				cvDrawContours(maskTemp, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8); // Only process up to *num of them

				if (centers != NULL) {				// Find the center of each contour
					cvMoments(maskTemp, &moments, 1);
					M00 = cvGetSpatialMoment(&moments, 0, 0);
					M10 = cvGetSpatialMoment(&moments, 1, 0);
					M01 = cvGetSpatialMoment(&moments, 0, 1);
					centers[i].x = (int)(M10 / M00);
					centers[i].y = (int)(M01 / M00);
				}
				if (bbs != NULL) {					//Bounding rectangles around blobs
					bbs[i] = cvBoundingRect(c);
				}
				cvZero(maskTemp);
				numFilled++;
			}

			cvDrawContours(mask, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8); // Draw filled contours into mask
		} //end looping over contours

		*num = numFilled;
		cvReleaseImage(&maskTemp);
	}
	/* Else just draw processed contours into the mask */
	else {
		// The user doesn!|t want statistics, just draw the contours
		for (c = contours; c != NULL; c = c->h_next) {
			cvDrawContours(mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
	}
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
	for (int y = max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if (fY >= foreground.rows)
			break;

		// start at the column indicated by location, 

		// or at column 0 if location.x is negative.
		for (int x = max(location.x, 0); x < background.cols; ++x)
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
					backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}
}

void MorphologyProcess(IplImage* &fgmaskIpl)
{
	//	static IplImage *dilateImg = 0, *erodeImg = 0, *maskMorphology = 0;
	//	maskMorphology = cvCloneImage(fgmaskIpl);
	//	erodeImg = cvCreateImage(cvSize(maskMorphology->width, maskMorphology->height), maskMorphology->depth, 1);
	//	dilateImg = cvCreateImage(cvSize(maskMorphology->width, maskMorphology->height), maskMorphology->depth, 1);
	//	int pos = 1;
	//	IplConvKernel * pKernel = NULL;
	//	pKernel = cvCreateStructuringElementEx(pos * 2 + 1, pos * 2 + 1, pos, pos, CV_SHAPE_ELLIPSE, NULL);
	//	for (int iter = 0; iter < 3; iter++){
	//		cvErode(maskMorphology, erodeImg, pKernel, 1);
	//		cvDilate(erodeImg, dilateImg, pKernel, 1);
	//	}	
	//	fgmaskIpl= cvCloneImage(dilateImg);
}

int Overlap(Rect a, Rect b, float ration)
{
	Rect c = a.x + a.width >= b.x + b.width ? a : b;
	Rect d = a.x + a.width >= b.x + b.width ? b : a;

	int e = min(d.x + d.width - c.x, d.width);
	if (e <= 0)
		return 0;

	c = a.y + a.height >= b.y + b.height ? a : b;
	d = a.y + a.height >= b.y + b.height ? b : a;

	int f = min(d.y + d.height - c.y, d.height);
	if (f <= 0)
		return 0;

	int overlapArea = e*f;
	int area_a = a.width * a.height;
	int area_b = b.width * b.height;
	int minArea = (area_a <= area_b ? area_a : area_b);

	if ((float)overlapArea / (float)minArea > ration) return 1;
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

void KF_init(cv::KalmanFilter *kf)
{
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
	unsigned int type = CV_32F;
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
		kf[i].processNoiseCov.at<float>(0) = 1e-2;
		kf[i].processNoiseCov.at<float>(7) = 1e-2;
		kf[i].processNoiseCov.at<float>(14) = 5.0f;
		kf[i].processNoiseCov.at<float>(21) = 5.0f;
		kf[i].processNoiseCov.at<float>(28) = 1e-2;
		kf[i].processNoiseCov.at<float>(35) = 1e-2;

		// Measures Noise Covariance Matrix R
		setIdentity(kf[i].measurementNoiseCov, Scalar(1e-1));
		// <<<< Kalman Filter
	}	
}


