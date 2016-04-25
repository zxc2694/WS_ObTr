#ifndef __KERNEL_H__
#define __KERNEL_H__

int parallel_similarity(double* hist, double* hist2, double &similarity);
void parallel_overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, int setting);
void parallel_computeHist(const Mat &roiMat, double *hist);
#endif