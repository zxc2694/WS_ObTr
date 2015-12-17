#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <stdio.h>

#define CVCONTOUR_APPROX_LEVEL	2
#define CVCLOSE_ITR				1	

void find_connected_components(IplImage *mask, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers)
{
	static CvMemStorage* mem_storage = NULL;
	static CvSeq* contours = NULL;

	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_OPEN, CVCLOSE_ITR);    //clear up raw mask
	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_CLOSE, CVCLOSE_ITR);

	/* find contours around only bigger regions */
	if (mem_storage == NULL) 
	{
		mem_storage = cvCreateMemStorage(0);
	}
	else
		cvClearMemStorage(mem_storage);

	CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	CvSeq* c;
	int numCont = 0;
	
	while ((c = cvFindNextContour(scanner)) != NULL) 
	{
		double len = cvContourPerimeter(c);
		double q = (mask->height + mask->width) / perimScale; // calculate perimeter len threshold

		/* Get rid of blob if its perimeter is too small: */
		if (len < q)
			cvSubstituteContour(scanner, NULL);

		else 
		{
			/* Smooth its edges if its large enough */
			CvSeq* c_new;
			if (poly1_hull0) {
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
	cvZero(mask);
	IplImage *maskTemp;
	
	/* Calc center of mass AND/OR bounding rectangles*/
	if (num != NULL) {		
		int N = *num, numFilled = 0, i = 0;
		CvMoments moments;
		double M00, M01, M10;
		maskTemp = cvCloneImage(mask);
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

			cvDrawContours(mask, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8); // Draw filled contours into mask
		} //end looping over contours

		*num = numFilled;
		cvReleaseImage(&maskTemp);
	}
	/* Else just draw processed contours into the mask */
	else {
		// The user doesn!|t want statistics, just draw the contours
		for (c = contours; c != NULL; c = c->h_next) {
			cvDrawContours(mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
	}
}