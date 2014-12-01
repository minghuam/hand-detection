/** @file cv_util.hpp
*	@brief Useful functions for OpenCV
*	@author minghuam
*/

#ifndef __CV_UTIL
#define __CV_UTIL

#include <string>
#include <opencv2/opencv.hpp>

/**
*	@brief Convert OpenCV data type into string.
*	@param type Opencv data type.
*/
static std::string CV_TYPE2STR(int type){
	std::string ret;
	int depth = type & CV_MAT_DEPTH_MASK;
	int ch = 1 + (type >> CV_CN_SHIFT);
	switch(depth){
	    case CV_8U:  ret = "8U"; break;
	    case CV_8S:  ret = "8S"; break;
	    case CV_16U: ret = "16U"; break;
	    case CV_16S: ret = "16S"; break;
	    case CV_32S: ret = "32S"; break;
	    case CV_32F: ret = "32F"; break;
	    case CV_64F: ret = "64F"; break;
	    default:     ret = "USER"; break;
	}

	ret += "C";
	ret += (ch + '0');
	return ret;
}

#endif /* __CV_UTIL */