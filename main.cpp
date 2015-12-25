#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "findComponent.h"
#include "codebook.h"
#include "ObjectTrackerFactory.h"
#include "math.h"
#include "MCFile.h"
#include <stdio.h>

using namespace std;
using namespace cv;
#define Pixel32S(img,x,y)	((int*)img.data)[(y)*img.cols + (x)]


/* Select images input */
#define Use_Webcam				0
#define Use_TestedVideo_Paul	1
#define Use_TestedVideo_Hardy	0

/* Save images output into the "video_output" file */
#define Save_imageOutput	1 

/* Select background subtrction algorithm */
#define Use_CodeBook	0
#define Use_MOG			1

/* Update the initial frame number of codebook */
int nframesToLearnBG = 1;  //if you use codebook, set 300. If you use MOG, set 1

/* Set tracking line length, range: 20~100 */
int plotLineLength = 50;

#define max(X, Y) (((X) >= (Y)) ? (X) : (Y))
#define min(X, Y) (((X) <= (Y)) ? (X) : (Y))

#define MAX_DIS_BET_PARTS_OF_ONE_OBJ		38

int Overlap(Rect a, Rect b, float ration)
{
	Rect c, d;
	if (a.x + a.width >= b.x + b.width)
	{
		c = a;
		d = b;
	}
	else
	{
		c = b;
		d = a;
	}
	int e = min(d.x + d.width - c.x, d.width);
	if (e <= 0)   return 0;

	if (a.y + a.height >= b.y + b.height)
	{
		c = a;
		d = b;
	}
	else
	{
		c = b;
		d = a;
	}
	int f = min(d.y + d.height - c.y, d.height);
	if (f <= 0)   return 0;

	int overlapArea = e*f;

	int area_a = a.width * a.height;
	int area_b = b.width * b.height;

	int minArea = (area_a <= area_b ? area_a : area_b);

	if ((float)overlapArea / (float)minArea > ration) return 1;
	return 0;
}

int main(int argc, const char** argv)
{
	char link[512];
	char outFilePath[100];
	char outFilePath2[100];
	bool update_bg_model = true;
	char prevData = false;
	int c, n, iter, iter2, MaxObjNum, nframes = 0;
	int pre_data_X[10] = { 0 }, pre_data_Y[10] = { 0 };	//for tracking line
	int first_last_diff = 1;								//compare first number with last number 

	CvRect bbs[10];
	CvPoint centers[10];
	IplImage *fgmaskIpl = 0;
	IplImage* image = 0, *yuvImage = 0;					//yuvImage is for codebook method
	IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;
	Mat img, fgmask, fgimg, show_img;
	vector<Object2D> object_list;
	vector<Object2D> prev_object_list;
	Object2D object;

	auto ms_tracker = ObjectTrackerFactory::create("MeanShiftTracker");		//local variables.	
	memset((object).hist, 0, MaxHistBins*sizeof(int));	

	namedWindow("image", WINDOW_NORMAL);
	namedWindow("foreground mask", WINDOW_NORMAL);

#if Use_Webcam
	CvCapture* capture = 0;
	capture = cvCaptureFromCAM(0);
#endif

#if Use_MOG		
	BackgroundSubtractorMOG2 bg_model;					//(100, 3, 0.3, 5);
#endif

#if Use_CodeBook
	CodeBookInit();										//Codebook initial function
#endif

	while (1)
	{
#if Save_imageOutput
		sprintf(outFilePath, "video_output//%05d.png", nframes + 1);
		sprintf(outFilePath2, "video_output//m%05d.png", nframes + 1);
		//sprintf(outFilePath, "video3_output//%05d.png", nframes + 180);
		//sprintf(outFilePath2, "video3_output//m%05d.png", nframes + 180);
#endif

#if Use_TestedVideo_Paul
		//sprintf(link, "D://Myproject//VS_Project//TestedVideo//video_output_1216//%05d.png", nframes+1);
		sprintf(link, "D://Myproject//VS_Project//TestedVideo//video3//%05d.png", nframes + 199);
		img = cvLoadImage(link, 1);
#endif

#if Use_TestedVideo_Hardy
		//sprintf(link, "D://test//tracking test//tracking test//video3//%05d.png", nframes + 180);
		sprintf(link, "D://test//tracking test//tracking test//video//%05d.png", nframes + 1);
		img = cvLoadImage(link, 1);
#endif

#if Use_Webcam
		img = cvQueryFrame(capture);
#endif

#if Use_MOG		
		bg_model(img, fgmask, update_bg_model ? -1 : 0); //update the model
		fgmaskIpl = &IplImage(fgmask);
#endif

#if Use_CodeBook
		image = &IplImage(img);
		RunCodeBook(image, yuvImage, ImaskCodeBook, ImaskCodeBookCC, nframes);  //Run codebook function
		fgmaskIpl = cvCloneImage(ImaskCodeBook);
#endif

		if (img.empty())
			break;

		static Mat TrackingLine(img.rows, img.cols, CV_8UC4);
		TrackingLine = Scalar::all(0);
		
		if (nframes < nframesToLearnBG)
		{

		}
		else if (nframes == nframesToLearnBG)
		{
			MaxObjNum = 10;															//less than 10 objects  
			find_connected_components(fgmaskIpl, 1, 4, &MaxObjNum, bbs, centers);

			for (int iter = 0; iter < MaxObjNum; ++iter)
			{
				ms_tracker->addTrackedList(img, object_list, bbs[iter], 2);
			}
			ms_tracker->track(img, object_list);
		}
		else
		{
			MorphologyProcess(fgmaskIpl);	// Run morphology 

			/* find components ,and compute bbs information  */
			MaxObjNum = 10;															//less than 10 objects  
			find_connected_components(fgmaskIpl, 1, 4, &MaxObjNum, bbs, centers);

			/* Plot the rectangles background subtarction finds */
			//for (iter = 0; iter < MaxObjNum; iter++){
			//	rectangle(img, bbs[iter], Scalar(0, 0, 255), 2); 
			//}

			LARGE_INTEGER m_liPerfFreq = { 0 };
			QueryPerformanceFrequency(&m_liPerfFreq);

			// Get executing time 
			LARGE_INTEGER m_liPerfStart = { 0 };
			QueryPerformanceCounter(&m_liPerfStart);

			ms_tracker->track(img, object_list);

			// Get executing time 
			LARGE_INTEGER liPerfNow = { 0 };
			QueryPerformanceCounter(&liPerfNow);

			// Compute total needed time (millisecond)
			long decodeDulation = (((liPerfNow.QuadPart - m_liPerfStart.QuadPart) * 1000) / m_liPerfFreq.QuadPart);
			// print 
			cout << "tracking time = " << decodeDulation << "ms" << endl;


			if (nframes > nframesToLearnBG + 1) // Start to update the object tracking
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
							cout << Pixel32S(ms_tracker->DistMat, min(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No),
								max(object_list[replaceList[iter1]].No, object_list[replaceList[iter2]].No)) << endl;

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
						object_list.erase(object_list.begin() + replaceList[iter]);
					}
				}

				if (!Overlapping && addToList)		ms_tracker->addTrackedList(img, object_list, bbs[bbs_iter], 2); //No replace and add object list -> bbs convert boundingBox.

				vector<int>().swap(replaceList);
			}  // end of 1st for 

			//cout << "draw frame"<<nframes
			ms_tracker->drawTrackBox(img, object_list);

			/* plotting trajectory */
			for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++) //Set all first ROI
			{			
				if (prevData == true) //prevent plotting tracking line when previous tracking data is none.
				{
					// Plotting all the tracking lines
					first_last_diff = ms_tracker->drawTrackTrajectory(TrackingLine, object_list, obj_list_iter);
					
					// Removing the tracking box when it's motionless for a while 
					if (first_last_diff == 0)
					{
						object_list.erase(object_list.begin() + obj_list_iter);
						first_last_diff = 1;
					}
				}				
				// Get previous point in order to use line function. 
				pre_data_X[obj_list_iter] = 0.5 * object_list[obj_list_iter].boundingBox.width + (object_list[obj_list_iter].boundingBox.x);
				pre_data_Y[obj_list_iter] = 0.9 * object_list[obj_list_iter].boundingBox.height + (object_list[obj_list_iter].boundingBox.y);

				if (object_list[obj_list_iter].PtNumber == plotLineLength + 1) //Restarting count when count > plotLineLength number
					object_list[obj_list_iter].PtNumber = 0;	
					
				object_list[obj_list_iter].point[object_list[obj_list_iter].PtNumber] = Point(pre_data_X[obj_list_iter], pre_data_Y[obj_list_iter]); //Storage all of points on the array. 
				object_list[obj_list_iter].PtNumber++;
				object_list[obj_list_iter].PtCount++;
			
			}// end of plotting trajectory
			prevData = true;
		}
		nframes++;

		/* Merge 3-channel image (original) and 4-channel image (for tracking) */
		overlayImage(img, TrackingLine, show_img, cv::Point(0, 0));
		imshow("image", show_img);
		cvShowImage("foreground mask", fgmaskIpl);

		char k = (char)waitKey(10);
		if (k == 27) break;

#if Save_imageOutput
		imwrite(outFilePath, show_img);
		cvSaveImage(outFilePath2, fgmaskIpl);
#endif

	}
#if Use_Webcam
	cvReleaseCapture(&capture);
#endif

	return 0;
}

