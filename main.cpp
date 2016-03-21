#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "Tracking.h"
#include "BackGroundModel.h"
#include "math.h"
#include <stdio.h>

/* Select images input */
#define inputPath_Paul   1
#define inputPath_Hardy  0
#define EtronCamera      0

/* Select background subtrction algorithm */
#define Use_CodeBook  0
#define Use_MOG       0
#define Use_DPEigenbackgroundBGS 1

/* Display*/
#define display_bbsRectangle  0  

#if EtronCamera
#include "WiCameraFactory.h"  //for Etron camera
#endif

int main(int argc, const char** argv)
{
	char inputPath[512];
	char outputPath[100];
	char outputPath2[100];
	bool update_bg_model = true;
	int nframes = 0;
	double t = 0;
	IplImage *fgmaskIpl = 0;
	IplImage* image = 0, *yuvImage = 0;
	IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;
	Mat img, img_compress;
	Mat	img_bgsModel, fgmask, imgTracking;

	CvRect bbs[MAX_OBJ_NUM], bbsV2[MAX_OBJ_NUM];
	CvPoint centers[MAX_OBJ_NUM];
	int ObjNum;

	// Initialization of background subtractions
	BackgroundSubtractorMOG2 bg_model;
	IBGS *bgs = new DPEigenbackgroundBGS;
	CodeBookInit();

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

	while (1) // for every frame
	{
#if inputPath_Paul
		sprintf(inputPath, "D:\\Myproject\\VS_Project\\TestedVideo\\video_output_1216\\%05d.png", nframes + 1);
		//sprintf(inputPath, "D:\\input_img\\%d.png", nframes + 30);
		//sprintf(inputPath, "D:\\Myproject\\VS_Project\\TestedVideo\\20160115Image\\L\\%d_L_Image.png", nframes + 194); // Vertical movement
		//sprintf(inputPath, "D:\\Myproject\\VS_Project\\TestedVideo\\20160115Image\\L_1\\%d_L_Image.png", nframes + 3424); //3424 //4024
		//sprintf(inputPath, "D:\\Myproject\\VS_Project\\TestedVideo\\CodeBook_videoOutput\\video_output_original\\%05d.png", nframes + 1);
		img = cvLoadImage(inputPath, 1);
#endif

#if inputPath_Hardy
		//sprintf(inputPath, "D://tracking data//3//%05d.png", nframes + 180);
		sprintf(inputPath, "D://tracking data//4//%05d.png", nframes + 1);
		//sprintf(inputPath, "D://tracking data//20160111Image//R_two_man//Jan%08d_R_Image.png", nframes + 11164513);
		//sprintf(inputPath, "D://tracking data//video_output_original//%05d.png", nframes + 2);
		//sprintf(inputPath, "D://tracking data//video_output_original2//%05d.png", nframes + 1);

		img = cvLoadImage(inputPath, 1);
#endif

#if EtronCamera	
		cam->getFrame((uchar*)frame.data);             // Capture the image from EStereo
		frame(Rect(0, 30, 640, 360)).copyTo(L_SrcImg);
		frame(Rect(640, 30, 640, 360)).copyTo(R_SrcImg);
		L_SrcImg.copyTo(img);
#endif

		if (img.empty()) break;

#if Use_MOG		
		resize(img, img_compress, cv::Size(img.cols / imgCompressionScale, img.rows / imgCompressionScale)); // compress img to 1/imgCompressionScale to speed up background subtraction and FindConnectedComponents
		bg_model.operator()(img_compress, fgmask, -1); //update the model
		//bg_model(img, fgmask, update_bg_model ? -1 : 0); //update the model
		imshow("fg", fgmask);
#endif

#if Use_CodeBook	
		resize(img, img_compress, cv::Size(img.cols / imgCompressionScale, img.rows / imgCompressionScale)); // compress img to 1/imgCompressionScale to speed up background subtraction and FindConnectedComponents
		image = &IplImage(img_compress);
		RunCodeBook(image, yuvImage, ImaskCodeBook, ImaskCodeBookCC, nframes);  //Run codebook function
		fgmaskIpl = cvCloneImage(ImaskCodeBook);
		fgmask = Mat(fgmaskIpl);
#endif

#if Use_DPEigenbackgroundBGS
		resize(img, img_compress, cv::Size(img.cols / imgCompressionScale, img.rows / imgCompressionScale)); // compress img to 1/imgCompressionScale to speed up background subtraction and FindConnectedComponents
		bgs->process(img_compress, fgmask, img_bgsModel);
#endif

		static FindConnectedComponents bbsFinder(img.cols, img.rows, imgCompressionScale, connectedComponentPerimeterScale);

		/* Get ROI */
		IplImage *fgmaskIpl = &IplImage(fgmask);

		bbsFinder.returnBbs(fgmaskIpl, &ObjNum, bbs, centers, true);

		t = (double)cvGetTickCount();         // Get executing time 	


		/* Plot tracking rectangles and its trajectory */
		tracking_function(img, imgTracking, fgmaskIpl, bbs, centers, ObjNum);


		t = (double)cvGetTickCount() - t;
		cout << "tracking time = " << t / ((double)cvGetTickFrequency() *1000.) << "ms,	nframes = " << nframes << endl;

		// Plot the rectangles background subtarction finds
		if (display_bbsRectangle)
		{
			for (int iter = 0; iter < ObjNum; iter++)
				rectangle(imgTracking, bbs[iter], Scalar(0, 255, 255), 2);
		}

		// Show the number of the frame on the image
		stringstream textFrameNo;
		textFrameNo << nframes;
		putText(imgTracking, "Frame=" + textFrameNo.str(), Point(10, imgTracking.rows - 10), 1, 1, Scalar(0, 0, 255), 1); //Show the number of the frame on the picture

		// Show image output
		imshow("Tracking_image", imgTracking);
		imshow("foreground mask", fgmask);
		sprintf(outputPath, "video_output_tracking//%05d.png", nframes + 1);
		sprintf(outputPath2, "video_output_BS//%05d.png", nframes + 1);
		imwrite(outputPath, imgTracking);
		imwrite(outputPath2, fgmask);

		nframes++;
		char k = (char)waitKey(10);
		if (k == 27) break;
	} // end of while

	delete bgs;
	destroyAllWindows();
	return 0;
}