#ifndef __MOTIONDETECTION_H__
#define __MOTIONDETECTION_H__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

extern int nframesToLearnBG;   // Update the initial frame number

class ImageBase
{
public:
	ImageBase(IplImage* img = NULL) { imgp = img; m_bReleaseMemory = true; }
	~ImageBase();

	void ReleaseMemory(bool b) { m_bReleaseMemory = b; }

	IplImage* Ptr() { return imgp; }
	const IplImage* Ptr() const { return imgp; }

	void ReleaseImage()
	{
		cvReleaseImage(&imgp);
	}

	void operator=(IplImage* img)
	{
		imgp = img;
	}

	// copy-constructor
	ImageBase(const ImageBase& rhs)
	{
		// it is very inefficent if this copy-constructor is called
		assert(false);
	}

	// assignment operator
	ImageBase& operator=(const ImageBase& rhs)
	{
		// it is very inefficent if operator= is called
		assert(false);

		return *this;
	}

	virtual void Clear() = 0;

protected:
	IplImage* imgp;
	bool m_bReleaseMemory;
};

class RgbImage : public ImageBase
{
public:
//	RgbImage(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img)
	{
		imgp = img;
	}

	// channel-level access using image(row, col, channel)
	inline unsigned char& operator()(const int r, const int c, const int ch)
	{
		return (unsigned char &)imgp->imageData[r*imgp->widthStep + c*imgp->nChannels + ch];
	}

	inline const unsigned char& operator()(const int r, const int c, const int ch) const
	{
		return (unsigned char &)imgp->imageData[r*imgp->widthStep + c*imgp->nChannels + ch];
	}

};

class BwImage : public ImageBase
{
public:
//	BwImage(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img)
	{
		imgp = img;
	}

	// pixel-level access using image(row, col)
	inline unsigned char& operator()(const int r, const int c)
	{
		return (unsigned char &)imgp->imageData[r*imgp->widthStep + c];
	}

	inline unsigned char operator()(const int r, const int c) const
	{
		return (unsigned char)imgp->imageData[r*imgp->widthStep + c];
	}
};


class BgsParams
{
public:
	virtual ~BgsParams() {}

	virtual void SetFrameSize(unsigned int width, unsigned int height)
	{
		m_width = width;
		m_height = height;
		m_size = width*height;
	}

	unsigned int &Width() { return m_width; }
	unsigned int &Height() { return m_height; }
	unsigned int &Size() { return m_size; }

protected:
	unsigned int m_width;
	unsigned int m_height;
	unsigned int m_size;
};


class Bgs
{
public:
	static const int BACKGROUND = 0;
	static const int FOREGROUND = 255;

	virtual ~Bgs() {}

	// Initialize any data required by the BGS algorithm. Should be called once before calling
	// any of the following functions.
	virtual void Initalize(const BgsParams& param) = 0;

	// Initialize the background model. Typically, the background model is initialized using the first
	// frame of the incoming video stream, but alternatives are possible.
	virtual void InitModel(const RgbImage& data) = 0;

	// Subtract the current frame from the background model and produce a binary foreground mask using
	// both a low and high threshold value.
	virtual void Subtract(int frame_num, const RgbImage& data,
		BwImage& low_threshold_mask, BwImage& high_threshold_mask) = 0;

	// Update the background model. Only pixels set to background in update_mask are updated.
	virtual void Update(int frame_num, const RgbImage& data, const BwImage& update_mask) = 0;

	// Return the current background model.
	virtual RgbImage *Background() = 0;
};


// --- Parameters used by the Mean BGS algorithm ---
class EigenbackgroundParams : public BgsParams
{
public:
	float &LowThreshold() { return m_low_threshold; }
	float &HighThreshold() { return m_high_threshold; }

	int &HistorySize() { return m_history_size; }
	int &EmbeddedDim() { return m_dim; }

private:
	// A pixel will be classified as foreground if the squared distance of any
	// color channel is greater than the specified threshold
	float m_low_threshold;
	float m_high_threshold;

	int m_history_size;			// number frames used to create eigenspace
	int m_dim;					// eigenspace dimensionality
};

// --- Eigenbackground BGS algorithm ---
class Eigenbackground : public Bgs
{
public:
	Eigenbackground();
	~Eigenbackground();

	void Initalize(const BgsParams& param);

	void InitModel(const RgbImage& data);
	void Subtract(int frame_num, const RgbImage& data,
		BwImage& low_threshold_mask, BwImage& high_threshold_mask);
	void Update(int frame_num, const RgbImage& data, const BwImage& update_mask);

	RgbImage* Background() { return &m_background; }

private:
	void UpdateHistory(int frameNum, const RgbImage& newFrame);

	EigenbackgroundParams m_params;

	CvMat* m_pcaData;
	CvMat* m_pcaAvg;
	CvMat* m_eigenValues;
	CvMat* m_eigenVectors;

	RgbImage m_background;
};

class DPEigenbackgroundBGS
{
private:
	// DPEigenBGS parameter
	bool firstTime;
	long frameNumber;
	IplImage* frame;
	RgbImage frame_data;

	EigenbackgroundParams params;
	Eigenbackground bgs;
	BwImage lowThresholdMask;
	BwImage highThresholdMask;

	int threshold;
	int historySize;
	int embeddedDim;
	bool showOutput;

public:
	DPEigenbackgroundBGS();
	~DPEigenbackgroundBGS();

	void process(const cv::Mat &img_input, cv::Mat &img_output);


};

class CMotionDetection
{
public:
	CMotionDetection(int nType);
	~CMotionDetection();
	bool MotionDetectionProcessing(Mat InputImage);
	Mat OutputFMask();   // Foreground image output
	
	// Background subtraction class
	void RunCodebook(Mat InputImage);     // Codebook function
	void RunMOG(Mat InputImage);          // MOG function
	void RunDPEigen(Mat InputImage);      // DPEigenBGS function
	void RunCodeBook_MOG(Mat InputImage); // Codebook + MOG function for 3D disparity

	// Find connected components
	void returnBbs(Mat BS_input, int *num, CvRect *bbs, CvPoint *centers, bool ignoreTooSmallPerimeter);
	void returnBbs_delShadow(Mat BS_input, int *num, CvRect *bbs, CvPoint *centers);
	void Output2dROI(Mat BS_input, CvRect *bbs, int *num); // 2D ROI output
	void Output3dROI();

private:
	int nBackGroudModel;
	int nFrameCntr;
	bool BGModelReady;
	Mat img_compress;   // Image input After it is Compressd
	Mat FMask;          // Image output

	// Codebook parameter
	CvBGCodeBookModel* model;
	IplImage *fgmaskIpl;
	IplImage image;
	IplImage *yuvImage;
	IplImage *ImaskCodeBook;
	IplImage *ImaskCodeBookCC;

	// MOG parameter
	BackgroundSubtractorMOG2 bg_model;

	// DPEigenBGS parameter
	DPEigenbackgroundBGS bgs;

	// Find connected components
	CvPoint centers[10];
	int method_Poly1_Hull0;
	int minConnectedComponentPerimeter;     // ignores obj with too small perimeter 
	float connectedComponentPerimeterScale; // when compute obj bbs, ignore obj with perimeter < (imgWidth + imgHeight) / (imgCompressionScale * ConnectedComponentPerimeterScale)
	int CVCONTOUR_APPROX_LEVEL;             // bbs parameter   
	int CVCLOSE_ITR;
};

void DensityFilter(BwImage& image, BwImage& filtered, int minDensity, unsigned char fgValue);

#endif

