#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include "Tracking.h"
#include <iomanip> 
#include <windows.h>
#include <math.h>

int objNumArray[10];
int objNumArray_BS[10];
CvBGCodeBookModel* model = 0;
Scalar *ColorPtr;

/* Function: tracking_function
 * @param: img        - Image input(RGB)
 * @param: fgmask     - Image after processing of background subtraction
 * @param: nframes    - The number of executions 
 * @param: ROI        - Input all ROIs
 * @param: ObjNum     - The number of objects
 * Note: 
 * 1. Both ROI and ObjNum are NULL, the funciton will use fgmask to find all ROIs of objects.
 * 2. Both ROI and ObjNum are not NULL, the function will ignore fgmask image.
*/
void tracking_function(Mat &img, Mat &fgmask, int &nframes, CvRect *ROI, int ObjNum)
{
	Mat show_img;
	CvRect bbs[10], bbsV2[10];
	CvPoint centers[10];
	char outFilePath[100];
	char outFilePath2[100];
	int c, n, iter, iter2, MaxObjNum;
	int first_last_diff = 1;                                    //compare first number with last number 
	static vector<Object2D> object_list;
	static char prevData = false;
	static int pre_data_X[10] = { 0 }, pre_data_Y[10] = { 0 };  //for tracking line
	static IObjectTracker *ms_tracker = new MeanShiftTracker(img.cols, img.rows, minObjWidth_Ini_Scale, minObjHeight_Ini_Scale, stopTrackingObjWithTooSmallWidth_Scale, stopTrackingObjWithTooSmallHeight_Scale);
	static Mat background_BBS(img.rows, img.cols, CV_8UC1);
	static Mat TrackingLine(img.rows, img.cols, CV_8UC4);       // Normal: cols = 640, rows = 480
	static FindConnectedComponents bbsFinder(img.cols, img.rows, imgCompressionScale, connectedComponentPerimeterScale);
	static KalmanF KF;
	vector<cv::Rect> KFBox;
	
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
		KF.Init();
	}
	else if (nframes < nframesToLearnBG)
	{

	}
	else if (nframes == nframesToLearnBG)
	{
		/* find components ,and compute bbs information  */
		MaxObjNum = 10; // bbsFinder don't find more than MaxObjNum objects  
		bbsFinder.returnBbs(fgmaskIpl, &MaxObjNum, bbs, centers, true);

		for (int iter = 0; iter < MaxObjNum; ++iter)
		{
			// decompression bbs and centers in fgmaskIpl
			bbs[iter].x *= imgCompressionScale;
			bbs[iter].y *= imgCompressionScale;
			bbs[iter].width *= imgCompressionScale;
			bbs[iter].height *= imgCompressionScale;
			centers[iter].x *= imgCompressionScale;
			centers[iter].y *= imgCompressionScale;

			ms_tracker->addTrackedList(img, object_list, bbs[iter], 2);
		}
	}
	else // case of nframes < nframesToLearnBG
	{
		if (ObjNum == NULL)                                                      //If ObjNum is NULL, we need to find all ROIs.
		{
			/* find components ,and compute bbs information  */
			MaxObjNum = 10; // bbsFinder don't find more than MaxObjNum objects  
			bbsFinder.returnBbs(fgmaskIpl, &MaxObjNum, bbs, centers, true);

			Mat(fgmaskIpl).copyTo(background_BBS);                               // Copy fgmaskIpl to background_BBS
			static Mat srcROI[10];                                               // for shadow rectangle

			/* Eliminating people's shadow method */
			for (iter = 0; iter < MaxObjNum; iter++)
			{                            // Get all shadow rectangles named bbsV2
				bbsV2[iter].x = bbs[iter].x;
				bbsV2[iter].y = bbs[iter].y + bbs[iter].height * 0.75;
				bbsV2[iter].width = bbs[iter].width;
				bbsV2[iter].height = bbs[iter].height * 0.25;
				srcROI[iter] = background_BBS(Rect(bbsV2[iter].x, bbsV2[iter].y, bbsV2[iter].width, bbsV2[iter].height)); // srcROI is depended on the image of background_BBS
				srcROI[iter] = Scalar::all(0);                                  // Set srcROI as showing black color
			}
			IplImage *BBSIpl = &IplImage(background_BBS);
			bbsFinder.returnBbs(BBSIpl, &MaxObjNum, bbs, centers, false);  // Secondly, Run the function of searching components to get update of bbs
	

			for (iter = 0; iter < MaxObjNum; iter++)
				bbs[iter].height = bbs[iter].height + bbsV2[iter].height;       // Merge bbs and bbsV2 to get final ROI

			// decompression bbs and centers in fgmaskIpl
			for (int iter = 0; iter < MaxObjNum; ++iter)
			{
				bbs[iter].x *= imgCompressionScale;
				bbs[iter].y *= imgCompressionScale;
				bbs[iter].width *= imgCompressionScale;
				bbs[iter].height *= imgCompressionScale;

				centers[iter].x *= imgCompressionScale;
				centers[iter].y *= imgCompressionScale;
			}

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
			int iter;
			for (iter = 0; iter < MaxObjNum; iter++)
				bbs[iter] = ROI[iter];
		}
		ms_tracker->track(img, object_list);

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

				for (int iter = 0; iter < object_list.size(); ++iter)
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

						if (Pixel32S(ms_tracker->DistMat, MIN(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
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

				ms_tracker->updateObjBbs(img, object_list, bbs[bbs_iter], replaceList[objWithLongestDuration]);

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

			if (!Overlapping && addToList)
			{
				ms_tracker->addTrackedList(img, object_list, bbs[bbs_iter], 2); //No replace and add object list -> bbs convert boundingBox.
			}

			vector<int>().swap(replaceList);
		}  // end of 1st for 

		for (iter = 0; iter < 10; iter++)
			objNumArray_BS[iter] = objNumArray[iter]; // Copy array from objNumArray to objNumArray_BS

		BubbleSort(objNumArray_BS, 10);               // Let objNumArray_BS array execute bubble sort 

		ms_tracker->drawTrackBox(img, object_list);   // Draw all the track boxes and their numbers 

		/* Removing motionless tracking box  */
		int black = 0;
		for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
		{
			//Calculating how much black point in image of background subtracion.
			for (int point = 0; point < 9; point++)
			{
			    for (int i = 0; i < DELE_RECT_FRAMENO; i++)
			    {
					if (object_list[obj_list_iter].ComparePoint[point][i] != 0)
						break;
					else
						black++;
				}
			}
			/* Modify the size of the tracking box  */
			if (DELE_RECT_FRAMENO * 5 < black)
			{
				int bbsNumber = 0;
				for (int i = 0; i < MaxObjNum; i++)
				{
					if (Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.5f))
					{
						bbsNumber++;
					}
				}
				for (int i = 0; i < MaxObjNum; i++)
				{
					if ((Overlap(object_list[obj_list_iter].boundingBox, bbs[i], 0.5f)) && (object_list[obj_list_iter].boundingBox.width > 1.5 * bbs[i].width) && (bbsNumber == 1))
					{
						//Reset the scale of the tracking box.
						ms_tracker->updateObjBbs(img, object_list, bbs[i], obj_list_iter);
					}
				}
			}
			// When its central point is all black in image of background subtracion.	
			if (DELE_RECT_FRAMENO * 9 == black)
			{
				char bbsExistObj = false;
				for (int i = 0; i < MaxObjNum; i++)
				{
					// To find internal bbs of the tracking box
					if ((centers[i].x > object_list[obj_list_iter].boundingBox.x) && (centers[i].x < object_list[obj_list_iter].boundingBox.x + object_list[obj_list_iter].boundingBox.width) && (centers[i].y > object_list[obj_list_iter].boundingBox.y) &&( centers[i].y < object_list[obj_list_iter].boundingBox.y + object_list[obj_list_iter].boundingBox.height))
					{
						//Reset the scale of the tracking box.
						object_list[obj_list_iter].objScale = 1;
						object_list[obj_list_iter].boundingBox = bbs[i];
						object_list[obj_list_iter].initialBbsWidth = bbs[i].width;
						object_list[obj_list_iter].initialBbsHeight = bbs[i].height;

						bbsExistObj = true;
						break;
					}
				}
				// If internal bbs of rectangle has existed, don't directly remove. 
				if (bbsExistObj == false)
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
			if (object_list.size() == 0)//Prevent out of vector range
				break;
		}

		/* plotting trajectory */
		for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
		{
			if (prevData == true) //prevent plotting tracking line when previous tracking data is none.
			{			
				ms_tracker->drawTrackTrajectory(TrackingLine, object_list, obj_list_iter); // Plotting all the tracking lines			
			}

			// Get previous point in order to use line function. 
			pre_data_X[obj_list_iter] = 0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x);
			pre_data_Y[obj_list_iter] = 0.9 * object_list[obj_list_iter].boundingBox.height + (object_list[obj_list_iter].boundingBox.y);

			// Restarting count when count > plotLineLength number
			if (object_list[obj_list_iter].PtNumber == plotLineLength + 1)
				object_list[obj_list_iter].PtNumber = 0;

			object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber] = Point(pre_data_X[obj_list_iter], pre_data_Y[obj_list_iter]); //Storage all of points on the array. 

			// Restarting count when count > DELE_RECT_FRAMENO number
			if (object_list[obj_list_iter].cPtNumber == DELE_RECT_FRAMENO + 1)
				object_list[obj_list_iter].cPtNumber = 0;

			IplImage *cfgmaskIpl = fgmaskIpl;
			// Get the color of nine points from tracking box (white or black)
			ComparePoint_9(cfgmaskIpl, object_list, obj_list_iter,object_list[obj_list_iter].cPtNumber);
		
			object_list[obj_list_iter].PtNumber++;
			object_list[obj_list_iter].cPtNumber++;
			object_list[obj_list_iter].PtCount++;

		}// end of plotting trajectory
		prevData = true;

		/* Kalman Filter Function */
		KF.Predict(img, object_list, KFBox);
		KF.Update(object_list, KFBox);

	} // case of nframes < nframesToLearnBG


	/* Show the number of the frame on the image */
	stringstream textFrameNo;
	textFrameNo << nframes;
	putText(img, "Frame=" + textFrameNo.str(), Point(10, img.rows - 10), 1, 1, Scalar(0, 0, 255), 1); //Show the number of the frame on the picture

	/* Display image output */
	overlayImage(img, TrackingLine, show_img, cv::Point(0, 0)); // Merge 3-channel image and 4-channel image
	imshow("Tracking_image", show_img);
	//cvShowImage("foreground mask", fgmaskIpl);

	imwrite(outFilePath, show_img);
	cvSaveImage(outFilePath2, fgmaskIpl);

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

		return sqrt((double)(c*c + d*d));
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

	for (int j = 0; j < 10; j++)                     //Set nine point as 255 in bbs
		for (int i = 0; i < DELE_RECT_FRAMENO; i++)
		  obj.ComparePoint[j][i] = 255;

	getKernel(obj.kernel, kernel_type);

	Mat tempMat = img(obj.boundingBox);
	computeHist(tempMat, obj.kernel, obj.hist);

	object_list.push_back(obj);

	for (size_t iter = 0; iter < object_list.size(); ++iter)
	{
		if (obj.No < object_list[(int)iter].No)
			Pixel32S(DistMat, obj.No, object_list[(int)iter].No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);
		else if (obj.No > object_list[(int)iter].No)
			Pixel32S(DistMat, object_list[(int)iter].No, obj.No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);
		
		//cout << Pixel32S(DistMat, object_list[(int)iter].No, obj.No);
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
/*				for (iter = 0; iter < 10; iter++)
				{
					if (objNumArray_BS[c] == objNumArray[iter])
					{
						ss3 << iter + 1;
						break;
					}
				}
				object_list[c].color = *(ColorPtr + iter);
*/
                ss3 << object_list[c].No;
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

				CandBbs[scaleIter].x += shift_x;
				CandBbs[scaleIter].y += shift_y;

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
			shift_x = round(shift_x);
			shift_y = round(shift_y);

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
//				cout << "iter " << Mean_Shift_Iter << "   similarity" << similarity << endl;
				break;
			}

			// iterate at most Max_Mean_Shift_Iter times
			if (Mean_Shift_Iter == Max_Mean_Shift_Iter)
			{
//				cout << "iter " << Mean_Shift_Iter << "   similarity" << similarity << endl;
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

void FindConnectedComponents::returnBbs(IplImage *mask, int *num, CvRect *bbs, CvPoint *centers, bool ignoreTooSmallPerimeter)
{
	static CvMemStorage* mem_storage = NULL;
	static CvSeq* contours = NULL;

	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_OPEN, 1);    //clear up raw mask
	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_CLOSE, CVCLOSE_ITR);

	/* find contours around only bigger regions */
	if (mem_storage == NULL)
	{
		mem_storage = cvCreateMemStorage(0);
	}
	else	cvClearMemStorage(mem_storage);

	CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	CvSeq* c;
	int numCont = 0;

	while ((c = cvFindNextContour(scanner)) != NULL)
	{
		double len = cvContourPerimeter(c);



		/* Get rid of blob if its perimeter is too small: */
		if (len < minConnectedComponentPerimeter && ignoreTooSmallPerimeter == true)	cvSubstituteContour(scanner, NULL);



		else
		{
			/* Smooth its edges if its large enough */
			CvSeq* c_new;
			if (method_Poly1_Hull0 == 1) {
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

void ComparePoint_9(IplImage* &fgmaskIpl, vector<Object2D> &object_list, int obj_list_iter, int PtN)
{
	int x = object_list[obj_list_iter].boundingBox.x;
	int y = object_list[obj_list_iter].boundingBox.y;
	int w = object_list[obj_list_iter].boundingBox.width;
	int h =	object_list[obj_list_iter].boundingBox.height;
	object_list[obj_list_iter].ComparePoint[0][PtN] = cvGet2D(fgmaskIpl, (0.2f * h + y) / imgCompressionScale, (0.2f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[1][PtN] = cvGet2D(fgmaskIpl, (0.2f * h + y) / imgCompressionScale, (0.5f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[2][PtN] = cvGet2D(fgmaskIpl, (0.2f * h + y) / imgCompressionScale, (0.8f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[3][PtN] = cvGet2D(fgmaskIpl, (0.5f * h + y) / imgCompressionScale, (0.2f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[4][PtN] = cvGet2D(fgmaskIpl, (0.5f * h + y) / imgCompressionScale, (0.5f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[5][PtN] = cvGet2D(fgmaskIpl, (0.5f * h + y) / imgCompressionScale, (0.8f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[6][PtN] = cvGet2D(fgmaskIpl, (0.8f * h + y) / imgCompressionScale, (0.2f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[7][PtN] = cvGet2D(fgmaskIpl, (0.8f * h + y) / imgCompressionScale, (0.5f * w + x) / imgCompressionScale).val[0];
	object_list[obj_list_iter].ComparePoint[8][PtN] = cvGet2D(fgmaskIpl, (0.8f * h + y) / imgCompressionScale, (0.8f * w + x) / imgCompressionScale).val[0];
	// Note: cvGet2D(IplImage*, y, x)
}

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

	ticks = (double)cv::getTickCount();
	dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

}

void KalmanF::Predict(Mat &img, vector<Object2D> &object_list, vector<cv::Rect> &ballsBox)
{
	if (found)
	{
		for (int i = 0; i < object_list.size(); i++)
		{
			// >>>> Matrix A
			kf[i].transitionMatrix.at<float>(2) = dT;
			kf[i].transitionMatrix.at<float>(9) = dT;
			// <<<< Matrix A

			//cout << "dT:" << endl << dT << endl;
			state[i] = kf[i].predict();
			//cout << "State post:" << endl << state[i] << endl;

			cv::Rect predRect;
			predRect.width = state[i].at<float>(4);
			predRect.height = state[i].at<float>(5);
			predRect.x = state[i].at<float>(0) - predRect.width / 2;
			predRect.y = state[i].at<float>(1) - predRect.height / 2;

			cv::Point center;
			center.x = state[i].at<float>(0);
			center.y = state[i].at<float>(1);
			if ((predRect.x != 0) && (predRect.y != 0) && display_kalmanRectangle == true)
			{
				cv::circle(img, center, 2, CV_RGB(255, 0, 0), -1); // central point of red rectangle
				cv::rectangle(img, predRect, CV_RGB(255, 0, 0), 2); //red rectangle --> predict
			}
		}
	}
	for (int iter = 0; iter < object_list.size(); iter++)
		ballsBox.push_back(object_list[iter].boundingBox);
}

void KalmanF::Update(vector<Object2D> &object_list, vector<cv::Rect> &ballsBox)
{
	if (object_list.size() == 0)
	{
		notFoundCount++;
		if (notFoundCount >= 100)
			found = false;
	}
	else
	{
		notFoundCount = 0;
		for (int i = 0; i < object_list.size(); i++)
		{
			meas[i].at<float>(0) = ballsBox[i].x + ballsBox[i].width / 2;
			meas[i].at<float>(1) = ballsBox[i].y + ballsBox[i].height / 2;
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
}

void drawArrow(Mat img, CvPoint p, CvPoint q)
{                  
	double angle; angle = atan2((double)p.y - q.y, (double)p.x - q.x);                       //bevel angle of pq line
	double hypotenuse; hypotenuse = sqrt((p.y - q.y)*(p.y - q.y) + (p.y - q.y)*(p.x - q.x)); //length of pq line
	
	/*The length of the arrow becomes three times from the original length */
	q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
	q.y = (int)(p.y - 3 * hypotenuse * sin(angle));
	
	/* Plot mainline */
	line(img, p, q, Scalar(0,0,0), 3, 1, 0);
	
	/* Plot two short lines */
	p.x = (int)(q.x + 9 * cos(angle + PI / 4));
	p.y = (int)(q.y + 9 * sin(angle + PI / 4));
	line(img, p, q, Scalar(0, 0, 0), 3, 1, 0);
	p.x = (int)(q.x + 9 * cos(angle - PI / 4));
	p.y = (int)(q.y + 9 * sin(angle - PI / 4));
	line(img, p, q, Scalar(0, 0, 0), 3, 1, 0);
}