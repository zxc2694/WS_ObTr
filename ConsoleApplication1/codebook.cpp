#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>

#include "codebook.h"
#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"

using namespace std;
using namespace cv; 


extern int nframesToLearnBG;
CvBGCodeBookModel* model = 0;


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
	if (nframes < nframesToLearnBG )
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