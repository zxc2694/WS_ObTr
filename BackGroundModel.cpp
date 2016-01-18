#include "BackGroundModel.h"

CodeBook::CodeBook()
{
	m_model = cvCreateBGCodeBookModel();
	//Set color thresholds to default values
	m_model->modMin[0] = 3;
	m_model->modMin[1] = m_model->modMin[2] = 3;
	m_model->modMax[0] = 10;
	m_model->modMax[1] = m_model->modMax[2] = 10;
	m_model->cbBounds[0] = m_model->cbBounds[1] = m_model->cbBounds[2] = 10;
}

CodeBook::~CodeBook()
{
	cvReleaseBGCodeBookModel(&m_model);
}

bool CodeBook::BGUpdate(Mat Mat_L_Cam)
{
	m_L_Cam = Mat_L_Cam;
	cvBGCodeBookUpdate(m_model, &m_L_Cam);
	return true;
}

bool CodeBook::GetFGMask(Mat Mat_L_Cam, Mat &Mat_FG)
{
	m_L_Cam = Mat_L_Cam;
	m_FG_Mask = Mat_FG;
	cvBGCodeBookDiff(m_model, &m_L_Cam, &m_FG_Mask);
	return true;
}

void CodeBook::ClearStale()
{
	cvBGCodeBookClearStale(m_model, m_model->t / 2);
}

void CodeBook::DefaultPostProcess(Mat &Mat_FG)
{
	m_FG_Mask = Mat_FG;
	cvSegmentFGMask(&m_FG_Mask);
}

/* BGS Algorithms */
using namespace Algorithms::BackgroundSubtraction;

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

void DPEigenbackgroundBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;

	loadConfig();

	if (firstTime)
		saveConfig();

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

	if (showOutput)
		cv::imshow("Eigenbackground (Oliver)", foreground);

	foreground.copyTo(img_output);

	delete frame;
	firstTime = false;
	frameNumber++;
}

void DPEigenbackgroundBGS::saveConfig()
{
	CvFileStorage* fs = cvOpenFileStorage("./config/DPEigenbackgroundBGS.xml", 0, CV_STORAGE_WRITE);

	cvWriteInt(fs, "threshold", threshold);
	cvWriteInt(fs, "historySize", historySize);
	cvWriteInt(fs, "embeddedDim", embeddedDim);
	cvWriteInt(fs, "showOutput", showOutput);

	cvReleaseFileStorage(&fs);
}

void DPEigenbackgroundBGS::loadConfig()
{
	CvFileStorage* fs = cvOpenFileStorage("./config/DPEigenbackgroundBGS.xml", 0, CV_STORAGE_READ);

	threshold = cvReadIntByName(fs, 0, "threshold", 225);
	historySize = cvReadIntByName(fs, 0, "historySize", 20);
	embeddedDim = cvReadIntByName(fs, 0, "embeddedDim", 10);
	showOutput = cvReadIntByName(fs, 0, "showOutput", true);

	cvReleaseFileStorage(&fs);
}