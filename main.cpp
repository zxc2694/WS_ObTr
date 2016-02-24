#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "Tracking.h"
#include "MotionDetection.h"
#include "math.h"
#include <stdio.h>

/* Select images input */
#define inputPath_Paul   1
#define inputPath_Hardy  0
#define EtronCamera      0

#if EtronCamera
#include "WiCameraFactory.h"  //for Etron camera
#endif

int main(int argc, const char** argv)
{
	char inputPath[100];
	char outputPath[100], outputPath2[100];
	int nframes = 0, ObjNum;
	double t = 0;
	Mat img, imgCompress, fgmask, imgTracking;
	CvRect ROI[10];	
	InputObjInfo trigROI;

	/* Select BS algorithm */
	CMotionDetection BS(2);    //Parameter 0: CodeBook, 1: MOG, 2: DPEigenBGS, 3: CodeBook+MOG

	//trigger box position
	trigROI.boundingBox.x = 500;
	trigROI.boundingBox.y = 350;
	trigROI.boundingBox.width = 50;
	trigROI.boundingBox.height = 80;
	trigROI.bIsTrigger = false;

#if EtronCamera
	// Set the parameter for EStereo
	static int img_width = 1280;
	static int img_height = 480;
	static string camera_type = "EStereoCamera";

	// Initial EStereo
	auto cam = WiCameraFactory::create(camera_type, img_width, img_height);
	char *camera_name = cam->get_camera_name();
	printf("camera name: %s\n", camera_name);
	int ckey;
	Mat frame(img_height, img_width, CV_8UC3);
	Mat L_SrcImg, R_SrcImg, L_GrayImg, R_GrayImg;
	Mat DisparityMap;
#endif

	while (1)
	{
#if inputPath_Paul
		sprintf(inputPath, "D://Myproject//VS_Project//TestedVideo//video_output_1216//%05d.png", nframes + 1);
		//sprintf(inputPath, "D://Myproject//VS_Project//TestedVideo//20160115Image//L//%d_L_Image.png", nframes + 194);
		//sprintf(inputPath, "D://Myproject//VS_Project//TestedVideo//20160115Image//L_1//%d_L_Image.png", nframes + 3424); //3424 //4024
		//sprintf(inputPath, "D://Myproject//VS_Project//TestedVideo//CodeBook_videoOutput//video_output_original//%05d.png", nframes + 1);
		img = cvLoadImage(inputPath, 1);
#endif

#if inputPath_Hardy
		//sprintf(inputPath, "D://tracking data//3//%05d.png", nframes + 180);
		sprintf(inputPath, "D://tracking data//4//%05d.png", nframes + 1);
		//sprintf(inputPath, "D://tracking data//20160111Image//R_two_man//Jan%08d_R_Image.png", nframes + 11164513);
		img = cvLoadImage(link, 1);
#endif

#if EtronCamera	
		cam->getFrame((uchar*)frame.data);             // Capture the image from EStereo
		frame(Rect(0, 30, 640, 360)).copyTo(L_SrcImg);
		frame(Rect(640, 30, 640, 360)).copyTo(R_SrcImg);
		L_SrcImg.copyTo(img);	
#endif

		if (img.empty()) break;

		resize(img, imgCompress, cv::Size(img.cols * 0.5, img.rows * 0.5));

		if (BS.MotionDetectionProcessing(imgCompress) != true){} // Build background model		
		
		else  // Initial background model has finished	
		{
			fgmask = BS.OutputFMask();            // Get image output of background subtraction			
			
			BS.Output2dROI(fgmask, ROI, &ObjNum); // Get ROI detection	
			
			t = (double)cvGetTickCount();         // Get executing time 


			tracking_function(img, imgTracking, ROI, ObjNum, &trigROI); // Plot tracking rectangles and their trajectories
	
			
			t = (double)cvGetTickCount() - t;
			cout << "tracking time = " << t / ((double)cvGetTickFrequency() *1000.) << "ms,	nframes = " << nframes << endl; 
				
			// Show the number of the frame on the image
			stringstream textFrameNo;
			textFrameNo << nframes;
			putText(imgTracking, "Frame=" + textFrameNo.str(), Point(10, imgTracking.rows - 10), 1, 1, Scalar(0, 0, 255), 1); //Show the number of the frame on the picture

			//// Show prohibited area
			//if (demoMode == true)
			//{
			//	rectangle(imgTracking, trigROI.boundingBox, Scalar(125, 10, 255), 2);
			//	line(imgTracking, Point(trigROI.boundingBox.x, trigROI.boundingBox.y), Point(trigROI.boundingBox.x + trigROI.boundingBox.width, trigROI.boundingBox.y + trigROI.boundingBox.height), Scalar(125, 10, 255), 2);
			//	line(imgTracking, Point(trigROI.boundingBox.x + trigROI.boundingBox.width, trigROI.boundingBox.y), Point(trigROI.boundingBox.x, trigROI.boundingBox.y + trigROI.boundingBox.height), Scalar(125, 10, 255), 2);
			//}
			// Show image output
			imshow("Tracking_image", imgTracking);
			imshow("foreground mask", fgmask);
			sprintf(outputPath, "video_output_tracking//%05d.png", nframes + 1);
			sprintf(outputPath2, "video_output_BS//%05d.png", nframes + 1);
			imwrite(outputPath, imgTracking);
			imwrite(outputPath2, fgmask);			
		}

		nframes++;	
		char k = (char)waitKey(10);
		if (k == 27) break;

	} // end of while

	return 0;
}
