#pragma once
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;

class CodeBook
{
public:
	CodeBook();
	~CodeBook();
	bool BGUpdate(Mat Mat_L_Cam);
	bool GetFGMask(Mat Mat_L_Cam, Mat &Mat_FG);
	void ClearStale();
	void DefaultPostProcess(Mat &Mat_FG);

private:
	CvBGCodeBookModel* m_model;
	IplImage m_L_Cam;
	IplImage m_FG_Mask;
};


