#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include "Tracking.h"
#include <iomanip> 
#include <math.h>
#include "MotionDetection.h"

int nframesToLearnBG = 1; 

CMotionDetection::CMotionDetection(int nType)
{
	nBackGroudModel = nType;
	nFrameCntr = 0;
	BGModelReady = false;

	// 0 means Codebook // 3 means Codebook + MOG
	if ((nBackGroudModel == 0) || (nBackGroudModel == 3))
	{
		model = cvCreateBGCodeBookModel();
		
		//Set color thresholds to default values
		model->modMin[0] = 3;
		model->modMin[1] = model->modMin[2] = 3;
		model->modMax[0] = 10;
		model->modMax[1] = model->modMax[2] = 10;
		model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;
	}

	// 1 means MOG
	else if (nBackGroudModel == 1)
	{

	}

	// 2 means DPEigen
	else if (nBackGroudModel == 2)
	{

	}

	// Find connected components
	method_Poly1_Hull0 = 1;                 // Use Polygon algorithm if method_Poly1_Hull0 = 1, and use Hull algorithm if method_Poly1_Hull0 = 0
	connectedComponentPerimeterScale = 6.0; // when compute obj bbs, ignore obj with perimeter < (imgWidth + imgHeight) / (imgCompressionScale * ConnectedComponentPerimeterScale)
	CVCONTOUR_APPROX_LEVEL = 2;             // bbs parameter   
	CVCLOSE_ITR = 3;                        // number of Recursive times for computing bbs
}

CMotionDetection::~CMotionDetection()
{

}

bool CMotionDetection::MotionDetectionProcessing(Mat InputImage)
{ 
	// 0 means Codebook
	if(nBackGroudModel == 0)
	{
		RunCodebook(InputImage);
	}

	// 1 means MOG
	if (nBackGroudModel == 1)
	{
		RunMOG(InputImage);
	}

	// 2 means MOG
	if (nBackGroudModel == 2)
	{
		RunDPEigen(InputImage);
	}

	// 3 means Codebook + MOG
	if (nBackGroudModel == 3)
	{
		RunCodebook(InputImage);
	}

	nFrameCntr++;
	return BGModelReady;
}

void CMotionDetection::RunCodebook(Mat InputImage)
{
	if (nFrameCntr == 0)                                 // Initialization of codeBook mask
	{
		image = InputImage;
		yuvImage = cvCloneImage(&image);
		ImaskCodeBook = cvCreateImage(cvGetSize(&image), IPL_DEPTH_8U, 1);
		ImaskCodeBookCC = cvCreateImage(cvGetSize(&image), IPL_DEPTH_8U, 1);
		cvSet(ImaskCodeBook, cvScalar(255));
	}
	cvCvtColor(&image, yuvImage, CV_BGR2YCrCb);           //YUV For codebook method
	
	if (nFrameCntr < nframesToLearnBG)                    //This is where we build our background model
		cvBGCodeBookUpdate(model, yuvImage); 

	if (nFrameCntr == nframesToLearnBG)
		cvBGCodeBookClearStale(model, model->t / 2);
	
	if (nFrameCntr >= nframesToLearnBG)                   //Find the foreground
	{
		BGModelReady = true;	

		cvBGCodeBookDiff(model, yuvImage, ImaskCodeBook); // Find foreground by codebook method
		cvCopy(ImaskCodeBook, ImaskCodeBookCC);           // This part just to visualize bounding boxes and centers if desired
		cvSegmentFGMask(ImaskCodeBookCC);
	}

	if (BGModelReady == true)
	{
		fgmaskIpl = cvCloneImage(ImaskCodeBook);
		FMask = Mat(fgmaskIpl);
	}
}

void CMotionDetection::RunMOG(Mat InputImage)
{
	bg_model.operator()(InputImage, FMask, -1);
	
	if (nFrameCntr >= nframesToLearnBG)
	{
		BGModelReady = true;
	}
}


void CMotionDetection::RunDPEigen(Mat InputImage)
{
	bgs.process(InputImage, FMask);

	if (nFrameCntr >= nframesToLearnBG)
	{
		BGModelReady = true;
	}
}

void CMotionDetection::RunCodeBook_MOG(Mat InputImage)
{
	Mat temp;
	bg_model.operator()(InputImage, temp, -1);
	
	if (nFrameCntr == 0)                                 // Initialization of codeBook mask
	{
		image = temp;
		yuvImage = cvCloneImage(&image);
		ImaskCodeBook = cvCreateImage(cvGetSize(&image), IPL_DEPTH_8U, 1);
		ImaskCodeBookCC = cvCreateImage(cvGetSize(&image), IPL_DEPTH_8U, 1);
		cvSet(ImaskCodeBook, cvScalar(255));
	}
	cvCvtColor(&image, yuvImage, CV_BGR2YCrCb);           //YUV For codebook method

	if (nFrameCntr < nframesToLearnBG)                    //This is where we build our background model
		cvBGCodeBookUpdate(model, yuvImage);

	if (nFrameCntr == nframesToLearnBG)
		cvBGCodeBookClearStale(model, model->t / 2);

	if (nFrameCntr >= nframesToLearnBG)                   //Find the foreground
	{
		BGModelReady = true;

		cvBGCodeBookDiff(model, yuvImage, ImaskCodeBook); // Find foreground by codebook method
		cvCopy(ImaskCodeBook, ImaskCodeBookCC);           // This part just to visualize bounding boxes and centers if desired
		cvSegmentFGMask(ImaskCodeBookCC);
	}

	if (BGModelReady == true)
	{
		fgmaskIpl = cvCloneImage(ImaskCodeBook);
		FMask = Mat(fgmaskIpl);
	}
}

Mat CMotionDetection::OutputFMask()
{
	return FMask;
}

void CMotionDetection::Output2dROI(Mat BS_input, CvRect *bbs, int *num)
{
	*num = 10;
	returnBbs(BS_input, num, bbs, centers, true);      //find ROI components
	returnBbs_delShadow(BS_input, num, bbs, centers);  //find final ROI components after finishing processing of shadow
}

void CMotionDetection::Output3dROI()
{

}

void CMotionDetection::returnBbs(Mat BS_input, int *num, CvRect *bbs, CvPoint *centers, bool ignoreTooSmallPerimeter)
{
	static CvMemStorage* mem_storage = NULL;
	static CvSeq* contours = NULL;
	IplImage mask = BS_input;

	cvMorphologyEx(&mask, &mask, 0, 0, CV_MOP_OPEN, 1);    //clear up raw mask
	cvMorphologyEx(&mask, &mask, 0, 0, CV_MOP_CLOSE, CVCLOSE_ITR);

	/* find contours around only bigger regions */
	if (mem_storage == NULL)
	{
		mem_storage = cvCreateMemStorage(0);
	}
	else	cvClearMemStorage(mem_storage);

	CvContourScanner scanner = cvStartFindContours(&mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
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
	cvZero(&mask);
	IplImage *maskTemp;

	/* Calc center of mass AND/OR bounding rectangles*/
	if (num != NULL) {
		int N = *num, numFilled = 0, i = 0;
		CvMoments moments;
		double M00, M01, M10;
		maskTemp = cvCloneImage(&mask);
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

			cvDrawContours(&mask, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8); // Draw filled contours into mask
		} //end looping over contours

		*num = numFilled;
		cvReleaseImage(&maskTemp);
	}
	/* Else just draw processed contours into the mask */
	else {
		// The user doesn!|t want statistics, just draw the contours
		for (c = contours; c != NULL; c = c->h_next) {
			cvDrawContours(&mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
	}
}

void CMotionDetection::returnBbs_delShadow(Mat BS_input, int *num, CvRect *bbs, CvPoint *centers)
{
	int iter = 0;
	CvRect bbsV2[10];
	IplImage fgmaskIpl = BS_input;
	Mat background_BBS((BS_input.rows >> 1), (BS_input.cols >> 1), CV_8UC1);
	Mat(&fgmaskIpl).copyTo(background_BBS);
	static Mat srcROI[10];                                 // for rectangles of shadows

	/* Eliminating people's shadow method */
	for (iter = 0; iter < *num; iter++)
	{
		// Get all shadow rectangles named bbsV2
		bbsV2[iter].x = bbs[iter].x;
		bbsV2[iter].y = (int)(bbs[iter].y + bbs[iter].height * 0.75);
		bbsV2[iter].width = (int)(bbs[iter].width);
		bbsV2[iter].height = (int)(bbs[iter].height * 0.25);
		srcROI[iter] = background_BBS(Rect(bbsV2[iter].x, bbsV2[iter].y, bbsV2[iter].width, bbsV2[iter].height)); // srcROI is depended on the image of background_BBS
		srcROI[iter] = Scalar::all(0);                      // Set srcROI as showing black color
	}
	IplImage BBSIpl = background_BBS;
	int tempObjNum = 10;

	returnBbs(&BBSIpl, &tempObjNum, bbs, centers, false);  // Secondly, Run the function of searching components to get update of bbs

	if (tempObjNum == *num)                           // Prevent objects of bbs2 more than objects of bbs1 
	{
		for (iter = 0; iter < *num; iter++)
			bbs[iter].height = bbs[iter].height + bbsV2[iter].height; // Merge bbs and bbsV2 to get final ROI
	}
	//cvShowImage("123", &BBSIpl);  // Show BS image with finished shadow processing 
}

ImageBase::~ImageBase()
{
	if (imgp != NULL && m_bReleaseMemory)
		cvReleaseImage(&imgp);
	imgp = NULL;
}

void DensityFilter(BwImage& image, BwImage& filtered, int minDensity, unsigned char fgValue)
{
	for (int r = 1; r < image.Ptr()->height - 1; ++r)
	{
		for (int c = 1; c < image.Ptr()->width - 1; ++c)
		{
			int count = 0;
			if (image(r, c) == fgValue)
			{
				if (image(r - 1, c - 1) == fgValue)
					count++;
				if (image(r - 1, c) == fgValue)
					count++;
				if (image(r - 1, c + 1) == fgValue)
					count++;
				if (image(r, c - 1) == fgValue)
					count++;
				if (image(r, c + 1) == fgValue)
					count++;
				if (image(r + 1, c - 1) == fgValue)
					count++;
				if (image(r + 1, c) == fgValue)
					count++;
				if (image(r + 1, c + 1) == fgValue)
					count++;

				if (count < minDensity)
					filtered(r, c) = 0;
				else
					filtered(r, c) = fgValue;
			}
			else
			{
				filtered(r, c) = 0;
			}
		}
	}
}

Eigenbackground::Eigenbackground()
{
	m_pcaData = NULL;
	m_pcaAvg = NULL;
	m_eigenValues = NULL;
	m_eigenVectors = NULL;
}

Eigenbackground::~Eigenbackground()
{
	if (m_pcaData != NULL) cvReleaseMat(&m_pcaData);
	if (m_pcaAvg != NULL) cvReleaseMat(&m_pcaAvg);
	if (m_eigenValues != NULL) cvReleaseMat(&m_eigenValues);
	if (m_eigenVectors != NULL) cvReleaseMat(&m_eigenVectors);
}

void Eigenbackground::Initalize(const BgsParams& param)
{
	m_params = (EigenbackgroundParams&)param;

	m_background = cvCreateImage(cvSize(m_params.Width(), m_params.Height()), IPL_DEPTH_8U, 3);
	m_background.Clear();
}

void Eigenbackground::InitModel(const RgbImage& data)
{
	if (m_pcaData != NULL) cvReleaseMat(&m_pcaData);
	if (m_pcaAvg != NULL) cvReleaseMat(&m_pcaAvg);
	if (m_eigenValues != NULL) cvReleaseMat(&m_eigenValues);
	if (m_eigenVectors != NULL) cvReleaseMat(&m_eigenVectors);

	m_pcaData = cvCreateMat(m_params.HistorySize(), m_params.Size() * 3, CV_8UC1);

	m_background.Clear();
}

void Eigenbackground::Update(int frame_num, const RgbImage& data, const BwImage& update_mask)
{
	// the eigenbackground model is not updated (serious limitation!)
}

void Eigenbackground::Subtract(int frame_num, const RgbImage& data,
	BwImage& low_threshold_mask, BwImage& high_threshold_mask)
{
	// create eigenbackground
	if (frame_num == m_params.HistorySize())
	{
		// create the eigenspace
		m_pcaAvg = cvCreateMat(1, m_pcaData->cols, CV_32F);
		m_eigenValues = cvCreateMat(m_pcaData->rows, 1, CV_32F);
		m_eigenVectors = cvCreateMat(m_pcaData->rows, m_pcaData->cols, CV_32F);
		cvCalcPCA(m_pcaData, m_pcaAvg, m_eigenValues, m_eigenVectors, CV_PCA_DATA_AS_ROW);

		int index = 0;
		for (unsigned int r = 0; r < m_params.Height(); ++r)
		{
			for (unsigned int c = 0; c < m_params.Width(); ++c)
			{
				for (int ch = 0; ch < m_background.Ptr()->nChannels; ++ch)
				{
					m_background(r, c, 0) = static_cast<unsigned char>(cvmGet(m_pcaAvg, 0, index) + 0.5);
					index++;
				}
			}
		}
	}

	if (frame_num >= m_params.HistorySize())
	{
		// project new image into the eigenspace
		int w = data.Ptr()->width;
		int h = data.Ptr()->height;
		int ch = data.Ptr()->nChannels;
		CvMat* dataPt = cvCreateMat(1, w*h*ch, CV_8UC1);
		CvMat data_row;
		cvGetRow(dataPt, &data_row, 0);
		cvReshape(&data_row, &data_row, 3, data.Ptr()->height);
		cvCopy(data.Ptr(), &data_row);

		CvMat* proj = cvCreateMat(1, m_params.EmbeddedDim(), CV_32F);
		cvProjectPCA(dataPt, m_pcaAvg, m_eigenVectors, proj);

		// reconstruct point
		CvMat* result = cvCreateMat(1, m_pcaData->cols, CV_32F);
		cvBackProjectPCA(proj, m_pcaAvg, m_eigenVectors, result);

		// calculate Euclidean distance between new image and its eigenspace projection
		int index = 0;
		for (unsigned int r = 0; r < m_params.Height(); ++r)
		{
			for (unsigned int c = 0; c < m_params.Width(); ++c)
			{
				double dist = 0;
				bool bgLow = true;
				bool bgHigh = true;
				for (int ch = 0; ch < 3; ++ch)
				{
					dist = (data(r, c, ch) - cvmGet(result, 0, index))*(data(r, c, ch) - cvmGet(result, 0, index));
					if (dist > m_params.LowThreshold())
						bgLow = false;
					if (dist > m_params.HighThreshold())
						bgHigh = false;
					index++;
				}

				if (!bgLow)
				{
					low_threshold_mask(r, c) = FOREGROUND;
				}
				else
				{
					low_threshold_mask(r, c) = BACKGROUND;
				}

				if (!bgHigh)
				{
					high_threshold_mask(r, c) = FOREGROUND;
				}
				else
				{
					high_threshold_mask(r, c) = BACKGROUND;
				}
			}
		}

		cvReleaseMat(&result);
		cvReleaseMat(&proj);
		cvReleaseMat(&dataPt);
	}
	else
	{
		// set entire image to background since there is not enough information yet
		// to start performing background subtraction
		for (unsigned int r = 0; r < m_params.Height(); ++r)
		{
			for (unsigned int c = 0; c < m_params.Width(); ++c)
			{
				low_threshold_mask(r, c) = BACKGROUND;
				high_threshold_mask(r, c) = BACKGROUND;
			}
		}
	}

	UpdateHistory(frame_num, data);
}

void Eigenbackground::UpdateHistory(int frame_num, const RgbImage& new_frame)
{
	if (frame_num < m_params.HistorySize())
	{
		CvMat src_row;
		cvGetRow(m_pcaData, &src_row, frame_num);
		cvReshape(&src_row, &src_row, 3, new_frame.Ptr()->height);
		cvCopy(new_frame.Ptr(), &src_row);
	}
}

DPEigenbackgroundBGS::DPEigenbackgroundBGS() : firstTime(true), frameNumber(0), threshold(225), historySize(20), embeddedDim(10), showOutput(true)
{
	std::cout << "DPEigenbackgroundBGS()" << std::endl;
}

DPEigenbackgroundBGS::~DPEigenbackgroundBGS()
{
	std::cout << "~DPEigenbackgroundBGS()" << std::endl;
}

void DPEigenbackgroundBGS::process(const cv::Mat &img_input, cv::Mat &img_output)
{
	frame = new IplImage(img_input);

	if (firstTime)
		frame_data.ReleaseMemory(false);
	frame_data = frame;

	if (firstTime)
	{
		int width = img_input.size().width;
		int height = img_input.size().height;

		lowThresholdMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		lowThresholdMask.Ptr()->origin = IPL_ORIGIN_BL;

		highThresholdMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		highThresholdMask.Ptr()->origin = IPL_ORIGIN_BL;

		params.SetFrameSize(width, height);
		params.LowThreshold() = threshold; //15*15;
		params.HighThreshold() = 2 * params.LowThreshold();	// Note: high threshold is used by post-processing 
		//params.HistorySize() = 100;
		params.HistorySize() = historySize;
		//params.EmbeddedDim() = 20;
		params.EmbeddedDim() = embeddedDim;

		bgs.Initalize(params);
		bgs.InitModel(frame_data);
	}

	bgs.Subtract(frameNumber, frame_data, lowThresholdMask, highThresholdMask);
	lowThresholdMask.Clear();
	bgs.Update(frameNumber, frame_data, lowThresholdMask);

	cv::Mat foreground(highThresholdMask.Ptr());

	//	if (showOutput)
	//		cv::imshow("Eigenbackground (Oliver)", foreground);

	foreground.copyTo(img_output);

	delete frame;
	firstTime = false;
	frameNumber++;
}