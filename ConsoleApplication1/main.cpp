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

#define max(X, Y) (((X) >= (Y)) ? (X) : (Y))
#define min(X, Y) (((X) <= (Y)) ? (X) : (Y))

#define MAX_DIS_BET_PARTS_OF_ONE_OBJ		20


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
	int c, n, iter, iter2, MaxObjNum, nframes = 0;
	int pre_data_X[10] = { 0 }, pre_data_Y[10] = {0};	//for tracking line

	CvRect bbs[10];
	CvPoint centers[10];
	IplImage *fgmaskIpl = 0;
	IplImage *dilateImg = 0, *erodeImg = 0, *maskMorphology = 0;
	IplImage* image = 0, *yuvImage = 0;					//yuvImage is for codebook method
	IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;

	Mat img, fgmask, fgimg, show_img;
	
	vector<Object2D> object_list;
	vector<Object2D> prev_object_list;
	Rect FirstROI[10];

	auto ms_tracker = ObjectTrackerFactory::create("MeanShiftTracker");		//local variables.

	Object2D object;
	memset((object).hist, 0, MaxHistBins*sizeof(int));
	char plotTrajectory = false;
	char FirstROI_status = false;

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
		sprintf(outFilePath, "video_output//%05d.png", nframes);
		sprintf(outFilePath2, "video_output//m%05d.png", nframes);
#endif

#if Use_TestedVideo_Paul
		//sprintf(link, "D://Myproject//VS_Project//TestedVideo//video_output1_20151211//%05d.png", nframes + 1500);
		sprintf(link, "D://Myproject//VS_Project//TestedVideo//video3//%05d.png", nframes + 180);
		img = cvLoadImage(link, 1);
#endif
#if Use_TestedVideo_Hardy
		sprintf(link, "D://test//tracking test//tracking test//video3//%05d.png", nframes + 180);
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

		IplImage *imgIpl;
		imgIpl = &IplImage(img);

		static Mat TrackingLine(img.rows, img.cols, CV_8UC3);
		if (nframes == 0)
		{
			TrackingLine = Scalar::all(0);
		}

		else if (nframes < nframesToLearnBG)
		{

		}
		else if (nframes == nframesToLearnBG)
		{
			MaxObjNum = 10;															//less than 5 objects  
			find_connected_components(fgmaskIpl, 1, 4, &MaxObjNum, bbs, centers);

			for (int iter = 0; iter < MaxObjNum; ++iter)
			{
				ms_tracker->addTrackedList(img, object_list, bbs[iter], 2);
			}
			ms_tracker->track(img, object_list);
		}
		else
		{
			///* Do morphology */
			//maskMorphology = cvCloneImage(fgmaskIpl);
			//erodeImg = cvCreateImage(cvSize(maskMorphology->width, maskMorphology->height), maskMorphology->depth, 1);
			//dilateImg = cvCreateImage(cvSize(maskMorphology->width, maskMorphology->height), maskMorphology->depth, 1);
			//int pos = 1;
			//IplConvKernel * pKernel = NULL;
			//pKernel = cvCreateStructuringElementEx(pos * 2 + 1, pos * 2 + 1, pos, pos, CV_SHAPE_ELLIPSE, NULL);
			//for (iter = 0; iter < 3; iter++){
			//	cvErode(maskMorphology, erodeImg, pKernel, 1);
			//	cvDilate(erodeImg, dilateImg, pKernel, 1);
			//}	
			//fgmaskIpl = cvCloneImage(dilateImg);

			/* find components ,and compute bbs information  */
			MaxObjNum = 10;															//less than 5 objects  
			find_connected_components(fgmaskIpl, 1, 4, &MaxObjNum, bbs, centers);

			///* Plot the rectangles background subtarction finds */
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


			if (nframes > nframesToLearnBG + 1)
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
	
			ms_tracker->drawTrackBox(img, object_list);

			/* plotting trajectory */
			if (object_list.size() != NULL)
			{						
				for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++) //Set all first ROI
				{				
					FirstROI[obj_list_iter].x = object_list[obj_list_iter].boundingBox.x;
					FirstROI[obj_list_iter].y = object_list[obj_list_iter].boundingBox.y;
					FirstROI[obj_list_iter].width = object_list[obj_list_iter].boundingBox.width;
					FirstROI[obj_list_iter].height = object_list[obj_list_iter].boundingBox.height;
				}
				if (plotTrajectory == true)
				{
					for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
					{										
						if ((pre_data_Y[obj_list_iter] != NULL) && (pre_data_X[obj_list_iter] - (0.5 * FirstROI[obj_list_iter].width + object_list[obj_list_iter].boundingBox.x)) <= 80 && (pre_data_X[obj_list_iter] - (0.5 * FirstROI[obj_list_iter].width + object_list[obj_list_iter].boundingBox.x)) > -80)									//prevent the tracking line from plotting when pre_data is none.
						line(TrackingLine, Point(0.5 * FirstROI[obj_list_iter].width + (object_list[obj_list_iter].boundingBox.x)
							, 0.9 * FirstROI[obj_list_iter].height + (object_list[obj_list_iter].boundingBox.y))
							, Point(pre_data_X[obj_list_iter], pre_data_Y[obj_list_iter]), object_list[obj_list_iter].color, 3, 1, 0);
					}				
				}
				for (obj_list_iter = 0; obj_list_iter < object_list.size(); obj_list_iter++)
				{
					pre_data_X[obj_list_iter] = 0.5 * FirstROI[obj_list_iter].width + (object_list[obj_list_iter].boundingBox.x);
					pre_data_Y[obj_list_iter] = 0.9 * FirstROI[obj_list_iter].height + (object_list[obj_list_iter].boundingBox.y);
				}
				plotTrajectory = true;			
			} // end of plotting trajectory
		}
		nframes++;

		imshow("image", img + TrackingLine);
		cvShowImage("foreground mask", fgmaskIpl);


		char k = (char)waitKey(10);
		if (k == 27) break;
		if (k == ' ')
		{
			update_bg_model = !update_bg_model;
			if (update_bg_model)
				printf("Background update is on\n");
			else
				printf("Background update is off\n");
		}

#if Save_imageOutput
			imwrite(outFilePath, img + TrackingLine);
			cvSaveImage(outFilePath2, fgmaskIpl);
#endif
	
	}
#if Use_Webcam
		cvReleaseCapture(&capture);
#endif

	return 0;
}

