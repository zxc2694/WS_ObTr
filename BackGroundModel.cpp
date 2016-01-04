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