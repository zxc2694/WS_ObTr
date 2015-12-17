#include "MeanShiftTracker.h"
#include "GeometricFuncs.h"
#include <iostream>
#include <iomanip> 

using namespace std;
extern vector<Mat> TrackingLine;

MeanShiftTracker::MeanShiftTracker() :kernel_type(0), radius(1), bin_width(32)
{
	bins = 256 / bin_width;
	DistMat = Mat::zeros(MAX_OBJ_LIST_SIZE, MAX_OBJ_LIST_SIZE, CV_32SC1);
  	ColorMatrix[0] = Scalar(0, 0, 255);
	ColorMatrix[1] = Scalar(255, 0, 0);
	ColorMatrix[2] = Scalar(0, 255, 0);
	ColorMatrix[3] = Scalar(0, 255, 255);
	ColorMatrix[4] = Scalar(255, 0, 255);
	ColorMatrix[5] = Scalar(255, 255, 0);
	ColorMatrix[6] = Scalar(122, 0, 255);
	ColorMatrix[7] = Scalar(255, 122, 0);
	ColorMatrix[8] = Scalar(0, 255, 122);
	ColorMatrix[9] = Scalar(80, 255, 80);

}

int MeanShiftTracker::DistBetObj(Rect a, Rect b)
{
	int c, d;
	if ((a.x > b.x + b.width || b.x > a.x + a.width) || (a.y > b.y + b.height || b.y > a.y + a.height))
	{
		c = min(abs(a.x - (b.x + b.width)), abs((a.x + a.width) - b.x));
		d = min(abs(a.y - (b.y + b.height)), abs((a.y + a.height) - b.y));
		return sqrt((float)(c*c + d*d));
	}
	else return 0;
}

void MeanShiftTracker::addTrackedList(const Mat &img, vector<Object2D> &object_list, Rect bbs, short type)
{
	++count;

	Object2D obj;
	obj.No = count;
	//CvScalar Scalar = cvScalar(rand() % 256, rand() % 256, rand() % 256);
	//CvScalar Scalar[10] = { cvScalar(0, 0, 255), cvScalar(0, 255, 0), cvScalar(255, 0, 0), cvScalar(0, 0, 255), cvScalar(0, 0, 255), cvScalar(0, 0, 255), cvScalar(0, 0, 255), cvScalar(0, 0, 255) };
	/*= { Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(0, 255, 255), Scalar(255, 255, 0) };*/
	obj.color = ColorMatrix[count%10];
	obj.type = type;
	obj.status = 3;
	obj.boundingBox = bbs; 
	obj.xyz.z = 20;
	memset(obj.hist, 0, MaxHistBins*sizeof(int));

	Mat kernel(obj.boundingBox.height, obj.boundingBox.width, CV_32FC1);
	getParzenWindow(kernel, radius, kernel_type);
	Mat tempMat = img(obj.boundingBox);
	getDensityEstimate(tempMat, kernel, obj.hist);

    object_list.push_back(obj);	

	for (size_t iter = 0; iter < object_list.size(); ++iter)
	{ 
		if (obj.No < object_list[(int)iter].No)
			Pixel32S(DistMat, obj.No, object_list[(int)iter].No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);
		else if (obj.No > object_list[(int)iter].No) 
			Pixel32S(DistMat, object_list[(int)iter].No, obj.No) = DistBetObj(obj.boundingBox, object_list[(int)iter].boundingBox);

		cout << Pixel32S(DistMat, object_list[(int)iter].No, obj.No);
		cout << "";

	}
}

void MeanShiftTracker::updateObjBbs(const Mat &img, vector<Object2D> &object_list, Rect bbs, int idx)
{
	object_list[idx].boundingBox = bbs;
	memset(object_list[idx].hist, 0, MaxHistBins*sizeof(int));
	Mat kernel(object_list[idx].boundingBox.height, object_list[idx].boundingBox.width, CV_32FC1);
	getParzenWindow(kernel, radius, kernel_type);
	Mat tempMat = img(object_list[idx].boundingBox);
	getDensityEstimate(tempMat, kernel, object_list[idx].hist);
}

bool MeanShiftTracker::checkTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list)
{
	if (object_list.empty())	return false;

	for (size_t c = 0; c < object_list.size(); c++)
	{
			if (object_list[c].similar_val > Similar_Val_Threshold - 0.02)
			{
				//prev_object_list[c].status = 2;
				//prev_object_list[c].boundingBox = object_list[c].boundingBox;
				//prev_object_list[c].similar_val = object_list[c].similar_val;
			}
			else
			{
				object_list.erase(object_list.begin() + c);
				(Mat)TrackingLine[c] = Scalar::all(0);
				//prev_object_list.erase(prev_object_list.begin() + c);
				c--;
			}
	}

	//// hit check: delete element one by one from pre_tracked_list until check finished
	//for (size_t i = 0; i < object_list.size(); i++){
	//	for (size_t j = 0; j < prev_object_list.size(); j++){
	//		/*if (tracked_list[i].type != prev_tracked_list[j].type)
	//		continue;*/
	//		// intersect check
	//		bool bHit = testObjectIntersection(object_list[i], prev_object_list[j]);
	//		// check object distance
	//		//bool bEqualDepth = checkByDepth(tracked_list[i].xyz.z, prev_tracked_list[j].xyz.z);
	//		if (bHit){
	//			prev_object_list.erase(prev_object_list.begin() + j);
	//			if (prev_object_list.empty()) break;
	//			if (j == 0) j = -1;
	//			else j--;
	//			//break;
	//		}
	//	}
	//}
	//// miss detected objects: the left elements in the prev_tracked_list are missed detected objects, and then copy to tracked_list
	//for (size_t c = 0; c < prev_object_list.size(); c++){
	//	prev_object_list[c].status = 3; // missed
	//	object_list.push_back(prev_object_list[c]);
	//}
	//if (prev_object_list.empty()) // no missed object
	//{
	//	prev_object_list = object_list;
	//	return false;
	//}
	//else{
	//	prev_object_list.clear();
	//	prev_object_list = object_list;
	//	return true;
	//}
	return true;
}

bool MeanShiftTracker::updateTrackedList(vector<Object2D> &object_list, vector<Object2D> &prev_object_list)
{
	for (size_t c = 0; c < object_list.size(); c++)
	{
		if (object_list[c].status == 3)
		{
			bool bIntraSec = testIntraObjectIntersection(object_list, c);
			if (object_list[c].similar_val > Similar_Val_Threshold-0.02 && !bIntraSec)
			{
				prev_object_list[c].status = 2;
				prev_object_list[c].boundingBox = object_list[c].boundingBox;
				prev_object_list[c].similar_val = object_list[c].similar_val;
			}
			else
			{
				object_list.erase(object_list.begin() + c);
				prev_object_list.erase(prev_object_list.begin() + c);
				c--;
			}
		}
	}
	return 1;
}

void MeanShiftTracker::drawTrackBox(Mat &img, vector<Object2D> &object_list)
{
	for (size_t c = 0; c < object_list.size(); c++){
		//for (size_t c = 0; c < 1; c++){
		//if (object_list[c].status == 2){
			if (object_list[c].type == 1){
//				Scalar color(0, rand() % 128, 255);
				cv::rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);
				
				std::stringstream ss,ss1,ss2,ss3;
				ss << std::fixed << std::setprecision(2) << object_list[c].xyz.z;
				//cv::putText(img, "person:" + ss.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y - 8), 1, 1, ColorMatrix[c]);
				ss1 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.x;
				ss2 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.y;
				//cv::putText(img, "prob:" + ss1.str() + "," + ss2.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, ColorMatrix[c]);
				//cv::putText(img, "prob:" + ss1.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, ColorMatrix[c]);
				ss3 << object_list[c].No;
				cv::putText(img, ss3.str(), Point(object_list[c].boundingBox.x + object_list[c].boundingBox.width / 2 - 10, object_list[c].boundingBox.y + object_list[c].boundingBox.height / 2), 1, 3, object_list[c].color, 3);
			}
			if (object_list[c].type == 2){
				Scalar color(rand() % 128, 255, 0);
				cv::rectangle(img, object_list[c].boundingBox, object_list[c].color, 2);

				std::stringstream ss, ss1, ss2, ss3;
				ss << std::fixed << std::setprecision(2) << object_list[c].xyz.z;
				//cv::putText(img, "car:" + ss.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y - 8), 1, 1, ColorMatrix[c]); //object_list[c].color
				ss1 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.x;
				ss2 << std::fixed << std::setprecision(2) << object_list[c].boundingBox.y;
				//cv::putText(img, "prob:" + ss1.str() + "," + ss2.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, ColorMatrix[c]);
				ss3 << object_list[c].No;
				cv::putText(img, ss3.str(), Point(object_list[c].boundingBox.x + object_list[c].boundingBox.width / 2 - 10, object_list[c].boundingBox.y + object_list[c].boundingBox.height / 2), 1, 3, object_list[c].color, 3);
				/*ss1 << std::fixed << std::setprecision(2) << object_list[c].similar_val;
				cv::putText(img, "prob:" + ss1.str(), Point(object_list[c].boundingBox.x, object_list[c].boundingBox.y + 12), 1, 1, color);*/
			}
		//}
	}
}

int MeanShiftTracker::track(Mat &img, vector<Object2D> &object_list)
{
	for (size_t c = 0; c < object_list.size(); c++)
	{
		//if (object_list[c].status == 3)
		//{ // miss detected
			Rect Pos = object_list[c].boundingBox;
			Mat kernel(Pos.height, Pos.width, CV_32FC1);
			getParzenWindow(kernel, radius, kernel_type);
			//
			//Mat grad_x, grad_y; // gradient of kernel
			//cv::Sobel(kernel, grad_x, CV_32F, 1, 0, 3);
			//cv::Sobel(kernel, grad_y, CV_32F, 0, 1, 3);
			//grad_x = -grad_x;
			//grad_y = -grad_y;
			//
			Mat kernerX = (Mat_<float>(1, 3) << -1, 0, 1);
			Mat kernerY = (Mat_<float>(3, 1) << -1, 0, 1);
			Mat grad_x, grad_y; // gradient of kernel
			cv::filter2D(kernel, grad_x, CV_32F, kernerX); // grad_x is kernel's gradient wrt x-axis
			cv::filter2D(kernel, grad_y, CV_32F, kernerY); // grad_y is kernel's gradient wrt y-axis
			grad_x = -grad_x;
			grad_y = -grad_y;

			Mat tempMat = img(Pos);
			float hist[MaxHistBins] = { 0 };
			getDensityEstimate(tempMat, kernel, hist);
			Mat weight(Pos.height, Pos.width, CV_32FC1);
			object_list[c].similar_val = computeSimilarity(tempMat, kernel, object_list[c].hist, hist, weight);
			cout << object_list[c].similar_val << endl;
			int iters = 0;
			while (iters < Max_Iters && object_list[c].similar_val < Similar_Val_Threshold)
			{
				float num_x = 0;
				float num_y = 0;
				float den = 0;
				for (int i = 0; i < kernel.rows; i++){
					for (int j = 0; j < kernel.cols; j++){
						num_x += i*weight.at<float>(i, j)*grad_x.at<float>(i, j); // grad_x is kernel's gradient wrt x-axis
						num_y += j*weight.at<float>(i, j)*grad_y.at<float>(i, j); // grad_y is kernel's gradient wrt y-axis
						//double val = cv::norm(grad_x.at<float>(i, j), grad_y.at<float>(i, j));
						float val = sqrt((grad_x.at<float>(i, j) - grad_y.at<float>(i, j))*(grad_x.at<float>(i, j) - grad_y.at<float>(i, j)));
						den += weight.at<float>(i, j)*val;
					}
				}
				//
				den += 1E-05;
				float dx = num_x / den;
				float dy = num_y / den;
				Pos.x += cvRound(dx);
				Pos.y += cvRound(dy);
				if (Pos.x < 0 || Pos.x + Pos.width > img.cols - 1 || Pos.y < 0 || Pos.y + Pos.height>img.rows - 1){
					//object_list[c].status = 4; // track lossed
					break; // break while
				}
				tempMat = img(Pos);
				getDensityEstimate(tempMat, kernel, hist);
				object_list[c].similar_val = computeSimilarity(tempMat, kernel, object_list[c].hist, hist, weight);
				object_list[c].boundingBox = Pos;
				iters++;
				cout << iters << ":" << object_list[c].similar_val << "...";
			} //end of while
			cout << Pos << endl;
		//} // end of if
	} // end of for
	return 1;
}

void MeanShiftTracker::getParzenWindow(Mat &kernel, const int R, const int func_type)
{
	int H = kernel.rows;
	int W = kernel.cols;
	switch (func_type){
	case 0:
	{
		// Gaussian:  
		// sigma = x/3 as a gaussian is almost equal to 0 from 3 * sigma.
		float sig_w = (float(R*W) / 2.0f) / 3.0f;
		float sig_h = (float(R*H) / 2.0f) / 3.0f;
		float dev_w = sig_w*sig_w;
		float dev_h = sig_h*sig_h;
		for (int i = 0; i < H; i++){
			float yy = (i - .5f*H)*(i - .5f*H);
			for (int j = 0; j < W; j++){
				float xx = (j - .5f*W)*(j - .5f*W);
				kernel.at<float>(i, j) = exp(-.5f*(yy / dev_h + xx / dev_w));
			}
		}
		break;// Gaussian:  
	}
	case 1:
	{
		// Uniform:
		for (int i = 0; i < H; i++){
			for (int j = 0; j < W; j++){
				float HH = ((float(2 * i) / (float)H - 1.0f) / (float)R)*((float(2 * i) / (float)H - 1.0f) / (float)R);
				float WW = ((float(2 * j) / (float)W - 1.0f) / (float)R)*((float(2 * j) / (float)W - 1.0f) / (float)R);
				if (HH + WW <= 1)
					kernel.at<float>(i, j) = 1;
				else
					kernel.at<float>(i, j) = 0;
			}
		}
		break;
	}
	case 2:
	{
		// Epanechnikov:
		for (int i = 0; i < H; i++){
			for (int j = 0; j < W; j++){
				float RH2 = (float(2 * i) / float(R*H) - 1.0f / (float)R)*(float(2 * i) / float(R*H) - 1.0f / (float)R);
				float RW2 = (float(2 * j) / float(R*W) - 1.0f / (float)R)*(float(2 * j) / float(R*W) - 1.0f / (float)R);
				kernel.at<float>(i, j) = (1 - RH2 - RW2);
				if (kernel.at<float>(i, j) < 0)
					kernel.at<float>(i, j) = 0;
			}
		}
		break;
	}
	}
}

void MeanShiftTracker::getDensityEstimate(const Mat &roiMat, const Mat &kernel, float hist[])
{
	if (roiMat.data == NULL) return;
	float kernel_sum = 0;
	if (roiMat.channels() == 3){
		for (int i = 0; i < kernel.rows; i++){
			for (int j = 0; j < kernel.cols; j++){
				Vec3b bgr = roiMat.at<Vec3b>(i, j);
				int idx = (bgr.val[0] / bin_width)*bins*bins + (bgr.val[1] / bin_width)*bins + bgr.val[2] / bin_width;
				hist[idx] += kernel.at<float>(i, j);
				kernel_sum += kernel.at<float>(i, j);
			}
		}
		for (int i = 0; i < bins*bins*bins; i++){
			hist[i] /= kernel_sum;
		}
	}
	else{ // gray 
		for (int i = 0; i < kernel.rows; i++){
			for (int j = 0; j < kernel.cols; j++){
				int idx = roiMat.at<uchar>(i, j) / bin_width;
				hist[idx] += kernel.at<float>(i, j);
				kernel_sum += kernel.at<float>(i, j);
			}
		}
		for (int i = 0; i < bins; i++){
			hist[i] /= kernel_sum;
		}
	}
}

double MeanShiftTracker::computeSimilarity(const Mat &roiMat, const Mat &kernel, const float target[], const float candidate[], Mat &weight)
{
	if (roiMat.data == NULL) return -1;
	double similar_val = 0;
	if (roiMat.channels() >= 3){
		for (int i = 0; i < roiMat.rows; i++){
			for (int j = 0; j < roiMat.cols; j++){
				Vec3b bgr = roiMat.at<Vec3b>(i, j);
				int idx = (bgr.val[0] / bin_width)*bins*bins + (bgr.val[1] / bin_width)*bins + bgr.val[2] / bin_width;
				weight.at<float>(i, j) = sqrt((float)(target[idx]) / (float)(candidate[idx] + 1e-5));
				/*if ((float)candidate[idx] == 0)
				weight.at<float>(i, j) = sqrt((float)(target[idx] + 1e-5) / (float)(candidate[idx] + 1e-5));
				else
				weight.at<float>(i, j) = sqrt((float)target[idx] / (float)candidate[idx]);*/
				similar_val += weight.at<float>(i, j)*kernel.at<float>(i, j);
			}
		}
	}
	else{
		for (int i = 0; i < roiMat.rows; i++){
			for (int j = 0; j < roiMat.cols; j++){
				int idx = roiMat.at<uchar>(i, j) / bin_width;
				weight.at<float>(i, j) = sqrt((float)(target[idx] + 1) / (float)(candidate[idx] + 1));
				similar_val += weight.at<float>(i, j)*kernel.at<float>(i, j);
			}
		}
	}
	return similar_val / double(roiMat.rows*roiMat.cols);
}

bool MeanShiftTracker::testObjectIntersection(Object2D &obj1, Object2D &obj2)
{
	return testBoxIntersection(obj1.boundingBox.x, obj1.boundingBox.y, obj1.boundingBox.x + obj1.boundingBox.width - 1, obj1.boundingBox.y + obj1.boundingBox.height - 1,
		obj2.boundingBox.x, obj2.boundingBox.y, obj2.boundingBox.x + obj2.boundingBox.width - 1, obj2.boundingBox.y + obj2.boundingBox.height - 1);
}

bool MeanShiftTracker::testIntraObjectIntersection(vector<Object2D> &object_list, int cur_pos)
{
	bool bSection = false;
	for (size_t c = 0; c < object_list.size(); c++){
		if (c == cur_pos) continue;
		if (object_list[c].status == 3) continue; // avoid to delete two tracked objects
		bSection = testBoxIntersection(object_list[cur_pos].boundingBox.x, object_list[cur_pos].boundingBox.y, object_list[cur_pos].boundingBox.x + object_list[cur_pos].boundingBox.width - 1, object_list[cur_pos].boundingBox.y + object_list[cur_pos].boundingBox.height - 1,
			object_list[c].boundingBox.x, object_list[c].boundingBox.y, object_list[c].boundingBox.x + object_list[c].boundingBox.width - 1, object_list[c].boundingBox.y + object_list[c].boundingBox.height - 1);
		if (bSection)	break;
	}
	return bSection;
}