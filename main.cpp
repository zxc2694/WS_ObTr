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

// Select images input
#define inputPath   0
#define Use_Webcam  1

// Select background subtrction algorithm
#define Use_CodeBook   0
#define Use_MOG        1
#define Use_DPEigenBGS 0

// Display
#define display_bbsRectangle  0  

void imageShow(Mat &imgTracking, Mat &fgmask, CvRect *bbs, CvPoint *centers, int ObjNum, int nframes)
{
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

	// Save image output
	char outputPath[100];
	char outputPath2[100];
	sprintf(outputPath, "video_output_tracking//%05d.png", nframes + 1);
	sprintf(outputPath2, "video_output_BS//%05d.png", nframes + 1);
	imwrite(outputPath, imgTracking);
	imwrite(outputPath2, fgmask);
}

void imgSource(Mat &img, int frame)
{
#if Use_Webcam
	static CvCapture* capture = 0;
	if (frame < 1)
		capture = cvCaptureFromCAM(0);
	img = cvQueryFrame(capture);
#endif
#if inputPath
	char source[512];
	sprintf(source, "D:\\Myproject\\VS_Project\\video_output_1216\\%05d.png", frame + 1);
	img = cvLoadImage(source, 1);
#endif
}

int main(int argc, const char** argv)
{
	bool update_bg_model = true;
	int nframes = 0, ObjNum;
	IplImage *fgmaskIpl = 0, * image = 0, *yuvImage = 0, *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;
	Mat img, img_compress, img_bgsModel, fgmask, imgTracking;
	CvRect bbs[MAX_OBJ_NUM], bbsV2[MAX_OBJ_NUM];
	CvPoint centers[MAX_OBJ_NUM];

	// Initialization of background subtractions
	BackgroundSubtractorMOG2 bg_model;
	IBGS *bgs = new DPEigenbackgroundBGS;
	CodeBookInit();

	while (1) // main loop
	{
		// image source 
		imgSource(img, nframes);
		if (img.empty()) break;

		// Motion detection mode
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

#if Use_DPEigenBGS
		resize(img, img_compress, cv::Size(img.cols / imgCompressionScale, img.rows / imgCompressionScale)); // compress img to 1/imgCompressionScale to speed up background subtraction and FindConnectedComponents
		bgs->process(img_compress, fgmask, img_bgsModel);
#endif
		// Get ROI
		static FindConnectedComponents bbsFinder(img.cols, img.rows, imgCompressionScale, connectedComponentPerimeterScale);
		IplImage *fgmaskIpl = &IplImage(fgmask);
		bbsFinder.returnBbs(fgmaskIpl, &ObjNum, bbs, centers, true);

		// Plot tracking rectangles and its trajectory
		tracking_function(img, imgTracking, fgmaskIpl, bbs, centers, ObjNum);

		// image output and saving data
		imageShow(imgTracking, fgmask, bbs, centers, ObjNum, nframes);

		nframes++;
		char k = (char)waitKey(10);
		if (k == 27) break;
	} // end of while

	delete bgs;
	destroyAllWindows();
	return 0;
}