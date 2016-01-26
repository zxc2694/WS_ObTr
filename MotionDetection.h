#ifndef __MOTIONDETECTION_H__
#define __MOTIONDETECTION_H__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/tracking.hpp"

#define Threthold 3            // Update the initial frame number
#define imgCompressionScale 2  // Compress img to 1/imgCompressionScale to speed up background subtraction and FindConnectedComponents

class CMotionDetection
{
public:
	CMotionDetection(int nType);
	~CMotionDetection();
	bool MotionDetectionProcessing(Mat InputImage);
	Mat OutputFMask();
	
	// Background subtraction class
	void RunCodebook(Mat InputImage); // Codebook function
	void RunMOG(Mat InputImage);      // MOG function
	void RunDPEigen(Mat InputImage);  // DPEigenBGS function

private:
	int nBackGroudModel;
	int nFrameCntr;
	bool BGModelReady;
	Mat img_compress;           // Image input After it is Compressd
	Mat FMask;                  // Image output
	
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
	IBGS *bgs;
	Mat	img_bgsModel;
};

template <class T>
class ImageIterator
{
public:
	ImageIterator(IplImage* image, int x = 0, int y = 0, int dx = 0, int dy = 0) :
		i(x), j(y), i0(0)
	{
		data = reinterpret_cast<T*>(image->imageData);
		step = image->widthStep / sizeof(T);

		nl = image->height;
		if ((y + dy)>0 && (y + dy) < nl)
			nl = y + dy;

		if (y<0)
			j = 0;

		data += step*j;

		nc = image->width;
		if ((x + dx) > 0 && (x + dx) < nc)
			nc = x + dx;

		nc *= image->nChannels;
		if (x>0)
			i0 = x*image->nChannels;
		i = i0;

		nch = image->nChannels;
	}


	/* has next ? */
	bool operator!() const { return j < nl; }

	/* next pixel */
	ImageIterator& operator++()
	{
		i++;
		if (i >= nc)
		{
			i = i0;
			j++;
			data += step;
		}
		return *this;
	}

	ImageIterator& operator+=(int s)
	{
		i += s;
		if (i >= nc)
		{
			i = i0;
			j++;
			data += step;
		}
		return *this;
	}

	/* pixel access */
	T& operator*() { return data[i]; }

	const T operator*() const { return data[i]; }

	const T neighbor(int dx, int dy) const
	{
		return *(data + dy*step + i + dx);
	}

	T* operator&() const { return data + i; }

	/* current pixel coordinates */
	int column() const { return i / nch; }
	int line() const { return j; }

private:
	int i, i0, j;
	T* data;
	int step;
	int nl, nc;
	int nch;
};

// --- Constants --------------------------------------------------------------

const unsigned char NUM_CHANNELS = 3;

// --- Pixel Types ------------------------------------------------------------

class RgbPixel
{
public:
	RgbPixel() { ; }
	RgbPixel(unsigned char _r, unsigned char _g, unsigned char _b)
	{
		ch[0] = _r; ch[1] = _g; ch[2] = _b;
	}

	RgbPixel& operator=(const RgbPixel& rhs)
	{
		ch[0] = rhs.ch[0]; ch[1] = rhs.ch[1]; ch[2] = rhs.ch[2];
		return *this;
	}

	inline unsigned char& operator()(const int _ch)
	{
		return ch[_ch];
	}

	inline unsigned char operator()(const int _ch) const
	{
		return ch[_ch];
	}

	unsigned char ch[3];
};

class RgbPixelFloat
{
public:
	RgbPixelFloat() { ; }
	RgbPixelFloat(float _r, float _g, float _b)
	{
		ch[0] = _r; ch[1] = _g; ch[2] = _b;
	}

	RgbPixelFloat& operator=(const RgbPixelFloat& rhs)
	{
		ch[0] = rhs.ch[0]; ch[1] = rhs.ch[1]; ch[2] = rhs.ch[2];
		return *this;
	}

	inline float& operator()(const int _ch)
	{
		return ch[_ch];
	}

	inline float operator()(const int _ch) const
	{
		return ch[_ch];
	}

	float ch[3];
};

// --- Image Types ------------------------------------------------------------

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
	RgbImage(IplImage* img = NULL) : ImageBase(img) { ; }

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

	// RGB pixel-level access using image(row, col)
	inline RgbPixel& operator()(const int r, const int c)
	{
		return (RgbPixel &)imgp->imageData[r*imgp->widthStep + c*imgp->nChannels];
	}

	inline const RgbPixel& operator()(const int r, const int c) const
	{
		return (RgbPixel &)imgp->imageData[r*imgp->widthStep + c*imgp->nChannels];
	}
};

class RgbImageFloat : public ImageBase
{
public:
	RgbImageFloat(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img)
	{
		imgp = img;
	}

	// channel-level access using image(row, col, channel)
	inline float& operator()(const int r, const int c, const int ch)
	{
		return (float &)imgp->imageData[r*imgp->widthStep + (c*imgp->nChannels + ch)*sizeof(float)];
	}

	inline float operator()(const int r, const int c, const int ch) const
	{
		return (float)imgp->imageData[r*imgp->widthStep + (c*imgp->nChannels + ch)*sizeof(float)];
	}

	// RGB pixel-level access using image(row, col)
	inline RgbPixelFloat& operator()(const int r, const int c)
	{
		return (RgbPixelFloat &)imgp->imageData[r*imgp->widthStep + c*imgp->nChannels*sizeof(float)];
	}

	inline const RgbPixelFloat& operator()(const int r, const int c) const
	{
		return (RgbPixelFloat &)imgp->imageData[r*imgp->widthStep + c*imgp->nChannels*sizeof(float)];
	}
};

class BwImage : public ImageBase
{
public:
	BwImage(IplImage* img = NULL) : ImageBase(img) { ; }

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

class BwImageFloat : public ImageBase
{
public:
	BwImageFloat(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img)
	{
		imgp = img;
	}

	// pixel-level access using image(row, col)
	inline float& operator()(const int r, const int c)
	{
		return (float &)imgp->imageData[r*imgp->widthStep + c*sizeof(float)];
	}

	inline float operator()(const int r, const int c) const
	{
		return (float)imgp->imageData[r*imgp->widthStep + c*sizeof(float)];
	}
};

// --- Image Functions --------------------------------------------------------
void DensityFilter(BwImage& image, BwImage& filtered, int minDensity, unsigned char fgValue);

namespace Algorithms
{
	namespace BackgroundSubtraction
	{
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
	}
}

namespace Algorithms
{
	namespace BackgroundSubtraction
	{
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
	}
}

class IBGS
{
public:
	virtual void process(const cv::Mat &img_input, cv::Mat &img_foreground, cv::Mat &img_background) = 0;
	/*virtual void process(const cv::Mat &img_input, cv::Mat &img_foreground){
	process(img_input, img_foreground, cv::Mat());
	}*/
	virtual ~IBGS(){}

private:

};

namespace Algorithms
{
	namespace BackgroundSubtraction
	{
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
	}
}

using namespace Algorithms::BackgroundSubtraction;

class DPEigenbackgroundBGS : public IBGS
{
private:
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

	void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:

};

#endif

