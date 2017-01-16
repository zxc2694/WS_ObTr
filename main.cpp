#if 1
#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "ObjectTracking.h"
#include "motionDetection.h"
#include "math.h"
#include <stdio.h>

// Select images input
#define inputPath   1
#define Use_Webcam  0

// Select background subtrction algorithm
#define Use_CodeBook   0
#define Use_MOG        0
#define Use_DPEigenBGS 1

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
#endif

#if 0
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}
	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}
	// Get the path to your CSV.
	string fn_csv = string(argv[1]);
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// The following lines create an Fisherfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// If you just want to keep 10 Fisherfaces, then call
	// the factory method like this:
	//
	//      cv::createFisherFaceRecognizer(10);
	//
	// However it is not useful to discard Fisherfaces! Please
	// always try to use _all_ available Fisherfaces for
	// classification.
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0) and use _all_
	// Fisherfaces, then call it with:
	//
	//      cv::createFisherFaceRecognizer(0, 123.0);
	//
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	// Here is how to get the eigenvalues of this Eigenfaces model:
	Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getMat("eigenvectors");
	// Get the sample mean from the training data
	Mat mean = model->getMat("mean");
	// Display or save:
	if (argc == 2) {
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	}
	else {
		imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	}
	// Display or save the first, at most 16 Fisherfaces:
	for (int i = 0; i < min(16, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Bone colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
		// Display or save:
		if (argc == 2) {
			imshow(format("fisherface_%d", i), cgrayscale);
		}
		else {
			imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
		}
	}
	// Display or save the image reconstruction at some predefined steps:
	for (int num_component = 0; num_component < min(16, W.cols); num_component++) {
		// Slice the Fisherface from the model:
		Mat ev = W.col(num_component);
		Mat projection = subspaceProject(ev, mean, images[0].reshape(1, 1));
		Mat reconstruction = subspaceReconstruct(ev, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		// Display or save:
		if (argc == 2) {
			imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
		}
		else {
			imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
		}
	}
	// Display if we are not writing to an output folder:
	if (argc == 2) {
		waitKey(0);
	}
	return 0;
}
#endif