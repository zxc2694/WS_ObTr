#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/gpu/gpumat.hpp> 
#include <opencv2/gpu/gpu.hpp>
#include "math.h"

using namespace cv;
using namespace std;

__global__ void overlayImageKernel(const gpu::PtrStepSz<uchar> background, const gpu::PtrStepSz<uchar> foreground, gpu::PtrStepSz<uchar> output)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	double opacity = ((double)foreground.data[y * foreground.step + x * 4 + 3]) / 255.;

	for (int c = 0; opacity > 0 && c < 3; ++c)
	{
		unsigned char foregroundPx = foreground.data[y * foreground.step + x * 4 + c];
		unsigned char backgroundPx = background.data[y * background.step + x * 3 + c];
		(output).data[y* (output).step + 3*x + c] = (uchar)(backgroundPx * (1. - opacity) + foregroundPx * opacity);
	}
}

__global__ void overlayImage2Kernel(const unsigned char *foreground, unsigned char *output)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	double opacity = ((double)foreground[y * 2560 + x * 4 + 3]) / 255.;

	for (int c = 0; opacity > 0 && c < 3; ++c)
	{
		unsigned char foregroundPx = foreground[y * 2560 + x * 4 + c];
		unsigned char backgroundPx = output[y * 1920 + x * 3 + c];
		output[y * 1920 + 3 * x + c] = (uchar)(backgroundPx * (1. - opacity) + foregroundPx * opacity);
	}
}

__global__ void similarity_Kernel(double *a, double *b, double *outputArray)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int i = tid + bid * blockDim.x;
	__shared__ double s_data[1024], colorArray[4096];

	colorArray[i] = sqrt(a[i] * b[i]);

	s_data[tid] = colorArray[bid * 1024 + tid];
	__syncthreads();

	for (int i = 512; i > 0; i /= 2)
	{
		if (tid < i)
			s_data[tid] = s_data[tid] + s_data[tid + i];
		__syncthreads();
	}
	if (tid == 0)
	{
		outputArray[bid] = s_data[0];
	}
}

__global__ void computeHistKernel(const gpu::PtrStepSz<uchar> roiMat, double *kernel, double *histOutput2)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	__shared__ double histOutput[4096];
	//double kernel_sum;

	if (kernel[x + y * roiMat.cols] == 0){}
	else
	{
		int val0 = roiMat.data[y* roiMat.step + 3 * x];
		int val1 = roiMat.data[y* roiMat.step + 3 * x + 1];
		int val2 = roiMat.data[y* roiMat.step + 3 * x + 2];
		int idx = (val0 / 16) * 256 + (val1 / 16) * 16 + val2 / 16;
		histOutput[idx] = histOutput[idx] + kernel[x + y * roiMat.cols];
		//kernel_sum = kernel_sum + kernel[x + y * roiMat.cols];
	}
}

__global__ void MatrixMulKernel(int *a, int *b, int *c)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = x + y*blockDim.x * gridDim.x;
	int width = blockDim.x * gridDim.x;
	int Pvalue = 0;
	for (int k = 0; k < width; k++)
		Pvalue = Pvalue + a[y * width + k] * b[k * width + x];

	c[tid] = Pvalue;
}

int parallel_similarity(double* hist, double* hist2, double &similarity)
{
	similarity = 0.0;

	const int arraySize = 4096;
	double d[4];
	double *dev_a = 0;
	double *dev_b = 0;
	double *dev_d = 0;

	cudaMalloc((void**)&dev_a, arraySize * sizeof(double));
	cudaMalloc((void**)&dev_b, arraySize * sizeof(double));
	cudaMalloc((void**)&dev_d, 4 * sizeof(double));

	cudaMemcpy(dev_a, hist, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, hist2, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d, d, 4 * sizeof(double), cudaMemcpyHostToDevice);

	double t = 0.0;
	t = (double)cvGetTickCount();

	similarity_Kernel << <4, 1024 >> >(dev_a, dev_b, dev_d);

	t = (double)cvGetTickCount() - t;
	printf("similarity_Kernel = %gms\n", t / ((double)cvGetTickFrequency() *1000.));

	cudaMemcpy(d, dev_d, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 4; i++)
	{
		similarity += d[i];
	}
	return 0;
}

void parallel_overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, int setting)
{
	background.copyTo(output);

	if (setting == 1) // Use "upload & download" API
	{
		gpu::GpuMat gpu_background, gpu_foreground, gpu_output;
		gpu_background.upload(background);
		gpu_output.upload(output);
		gpu_foreground.upload(foreground);

		dim3 blocks(1, background.rows);
		dim3 threads(background.cols, 1);
		overlayImageKernel << < blocks, threads >> > (gpu_background, gpu_foreground, gpu_output);

		gpu_output.download(output);
	}
	if (setting == 2) // Use "cudaMemcpy"
	{
		unsigned char *dev_foreground = 0;
		unsigned char *dev_output = 0;
		const int b_Size = background.cols * background.rows * 3;
		const int f_Size = background.cols * background.rows * 4;

		cudaMalloc((void**)&dev_foreground, f_Size * sizeof(unsigned char));
		cudaMalloc((void**)&dev_output, b_Size * sizeof(unsigned char));
		cudaMemcpy(dev_foreground, foreground.data, f_Size * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_output, output.data, b_Size * sizeof(unsigned char), cudaMemcpyHostToDevice);

		dim3 blocks(1, background.rows);
		dim3 threads(background.cols, 1);
		overlayImage2Kernel << < blocks, threads >> > (dev_foreground, dev_output);

		cudaMemcpy(output.data, dev_output, b_Size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}
}

void parallel_computeHist(const Mat &roiMat, double *hist)
{
	double kernel[20000];
	int H = roiMat.rows - 1;
	int W = roiMat.cols - 1;
	int w = W / 2, h = H / 2;
	int kernelSize = roiMat.rows * roiMat.cols;
	double histTest[4096];
	memset(hist, 0, 4096 * sizeof(double)); // reset hist to 0
	memset(kernel, 0, 20000 * sizeof(double)); // reset hist to 0
	double kernel_sum = 0; // sum for normalize

	if (roiMat.data == NULL) return;
	int n = 0;
	for (int y = 0; y < roiMat.rows; y++)
	{
		for (int x = 0; x < roiMat.cols; x++)
		{
			// scale to unit circle
			float dist_y = (float)(y - h) / h;
			float dist_x = (float)(x - w) / w;
			float distToCen = dist_x*dist_x + dist_y*dist_y; // distance from (i, j) to bbs center

			if (distToCen >= 1)
				kernel[x + y * roiMat.cols] = 0.0;
			else
				kernel[x + y * roiMat.cols] = 2 * (1 - distToCen) * 0.31831;

			if (kernel[x + y * roiMat.cols] == 0){}
			else
			{
				int val0 = roiMat.data[y* roiMat.step + 3 * x];
				int val1 = roiMat.data[y* roiMat.step + 3 * x + 1];
				int val2 = roiMat.data[y* roiMat.step + 3 * x + 2];
				int idx = (val0 / 16) * 256 + (val1 / 16) * 16 + val2 / 16;
				hist[idx] += kernel[x + y * roiMat.cols];
				kernel_sum += kernel[x + y * roiMat.cols];
			}
		}
	}

/*	gpu::GpuMat gpu_roiMat;
	gpu_roiMat.upload(roiMat);

	double *dev_h = 0;
	cudaMalloc((void**)&dev_h, 4096 * sizeof(double));
	cudaMemcpy(dev_h, hist, 4096 * sizeof(double), cudaMemcpyHostToDevice);

	double *dev_k = 0;
	cudaMalloc((void**)&dev_k, kernelSize * sizeof(double));
	cudaMemcpy(dev_k, kernel, kernelSize * sizeof(double), cudaMemcpyHostToDevice);

	dim3 blocks(1, roiMat.rows);
	dim3 threads(roiMat.cols, 1);
	computeHistKernel << < blocks, threads >> > (gpu_roiMat, dev_k, dev_h);

	cudaMemcpy(hist, dev_h, 4096 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(kernel, dev_k, 4096 * sizeof(double), cudaMemcpyDeviceToHost);*/

	for (int i = 0; i < 4096;i++)
	{
		histTest[i] = hist[i];
	}
	for (int i = 0; i < 4096; i++)
	{
		hist[i] /= kernel_sum;
	}
}

