#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "Tracking.h"
#include "math.h"
#include <stdio.h>
#include <windows.h>

using namespace std;
using namespace cv;

/* Select images input */
#define inputPath_Paul   1
#define inputPath_Hardy  0

/* Select background subtrction algorithm */
#define Use_CodeBook  0
#define Use_MOG       1

int main(int argc, const char** argv)
{
	char link[512];
	char outFilePath[100];
	char outFilePath2[100];
	bool update_bg_model = true;
	int nframes = 0;
	IplImage *fgmaskIpl = 0;
	IplImage* image = 0, *yuvImage = 0;                  
	IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;
	Mat img, fgmask;
	Object2D object;
	BackgroundSubtractorMOG2 bg_model;
	CodeBookInit();
		
	IObjectTracker *ms_tracker = new MeanShiftTracker();
	memset((object).hist, 0, MaxHistBins*sizeof(int));
	
	while (1)
	{

#if inputPath_Paul
		//sprintf(link, "D://Myproject//VS_Project//TestedVideo//video_output_1216//%05d.png", nframes+1);
		sprintf(link, "D://Myproject//VS_Project//TestedVideo//video3//%05d.png", nframes + 180);
		img = cvLoadImage(link, 1);
#endif

#if inputPath_Hardy
		//sprintf(link, "D://test//tracking test//tracking test//video3//%05d.png", nframes + 180);
		sprintf(link, "D://test//tracking test//tracking test//video//%05d.png", nframes + 1);
		img = cvLoadImage(link, 1);
#endif

#if Use_MOG		
		bg_model(img, fgmask, update_bg_model ? -1 : 0); //update the model
#endif

#if Use_CodeBook
		image = &IplImage(img);
		RunCodeBook(image, yuvImage, ImaskCodeBook, ImaskCodeBookCC, nframes);  //Run codebook function
		fgmaskIpl = cvCloneImage(ImaskCodeBook);
		fgmask = Mat(fgmaskIpl);
#endif

		if (img.empty()) break;
	
		/* Plot tracking rectangles and its trajectory */
		tracking_function(img, fgmask, ms_tracker, object, nframes);

		nframes++;	
		char k = (char)waitKey(10);
		if (k == 27) break;

	}
	return 0;
}
