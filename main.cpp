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
	char link[512];
	int nframes = 0;
	double t = 0;
	Mat img;
	Mat	EXEFMask;

	/* Select BS algorithm */
	CMotionDetection BS(0);    //Parameter 0: CodeBook, 1: MOG, 2: DPEigenBGS

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
		//sprintf(link, "D://Myproject//VS_Project//TestedVideo//video_output_1216//%05d.png", nframes+1);
		//sprintf(link, "D://Myproject//VS_Project//TestedVideo//20160115Image//L//%d_L_Image.png", nframes + 194);
		//sprintf(link, "D://Myproject//VS_Project//TestedVideo//20160115Image//L_1//%d_L_Image.png", nframes + 3424); //3424 //4024
		sprintf(link, "D://Myproject//VS_Project//TestedVideo//CodeBook_videoOutput//video_output_original//%05d.png", nframes + 1);
		img = cvLoadImage(link, 1);
#endif

#if inputPath_Hardy
		//sprintf(link, "D://tracking data//3//%05d.png", nframes + 180);
		sprintf(link, "D://tracking data//4//%05d.png", nframes + 1);
		//sprintf(link, "D://tracking data//20160111Image//R_two_man//Jan%08d_R_Image.png", nframes + 11164513);
		img = cvLoadImage(link, 1);
#endif

#if EtronCamera	
		cam->getFrame((uchar*)frame.data);             // Capture the image from EStereo
		frame(Rect(0, 30, 640, 360)).copyTo(L_SrcImg);
		frame(Rect(640, 30, 640, 360)).copyTo(R_SrcImg);
		L_SrcImg.copyTo(img);	
#endif

		if (img.empty()) break;

		if (BS.MotionDetectionProcessing(img) != true){} // Background model is finished while MotionDetectionProcessing() is true
		
		else
		{
			EXEFMask = BS.OutputFMask();

			t = (double)cvGetTickCount(); // Get executing time 

			/* Plot tracking rectangles and its trajectory */
			tracking_function(img, EXEFMask, nframes, NULL, NULL);

			t = (double)cvGetTickCount() - t;
			cout << "tracking time = " << t / ((double)cvGetTickFrequency() *1000.) << "ms" << endl;
		}

		nframes++;	
		char k = (char)waitKey(10);
		if (k == 27) break;

	} // end of while

	return 0;
}
