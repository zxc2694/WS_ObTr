#ifndef GEOMETRICFUNCS_H
#define GEOMETRICFUNCS_H

bool testBoxIntersection(int left1, int top1, int right1, int bottom1, int left2, int top2, int right2, int bottom2);


bool testBoxIntersection(int left1, int top1, int right1, int bottom1, int left2, int top2, int right2, int bottom2)
{
	if (right1 < left2)	return false;	// 1 is left of 2
	if (left1 > right2) return false;	// 1 is right of 2
	if (bottom1 < top2)	return false;	// 1 is above of 2
	if (top1 > bottom2) return false;	// 1 is below of 2
	return true;
}
#endif 