/** @file color_feature.cpp
*	@brief ColorFeature class implementation file.
*	@author minghuam
*/

#include "color_feature.hpp"
#include <sstream>

/**
*	@brief Default constructor
*/
ColorFeature::ColorFeature(){
	ColorFeature(CS_RGB, 1);
}

/**
*	@brief Constructor
*	@param cs Color space
*	@param patch_size Patch size, must be odd.
*/
ColorFeature::ColorFeature(COLOR_SPACE cs, int patch_size) {
	_color_space = cs;
	_patch_size = patch_size;

	_cost = -1.0;
	_cache =NULL;

	if(_patch_size == 1){
		_dimension =  3;
	}else{
		_dimension = _patch_size * 2 + (_patch_size - 2) * 2;
		_dimension *= 3;
	}

	/* generate feature key */
	std::stringstream ss;
	ss << cs2str(_color_space) << "-" << patch_size;
	_key = ss.str();
}

/**
*	@brief Compute feature given an image and key points.
*	@param img Image input.
*	@param keypts Key points input.
*/
void ColorFeature::compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts){

	if(!keypts.size()){
		return;
	}
	
	double start_tick = cv::getTickCount();

	/* adjust descriptor size */
	if(_descriptor.rows != keypts.size() || _descriptor.cols != _dimension){
		_descriptor = cv::Mat::zeros(keypts.size(), _dimension, CV_32F);
	}

	/* color space conversion */
	cv::Mat color;
	if(_cache != NULL){
		if(_cache->get(cs2str(_color_space), color) == -1){
			covert_color(img, color);
			_cache->push(cs2str(_color_space), color);
		}
	}else{
		covert_color(img, color);
	}

	/* sampling */
	int win_size = _patch_size;
	for(int k = 0; k < (int)keypts.size(); k++)
	{
		int r = int(floor(.5 + keypts[k].pt.y) - floor(win_size*0.5));
		int c = int(floor(.5 + keypts[k].pt.x) - floor(win_size*0.5));
		int a = 0;
		for(int i = 0; i < win_size; i++)
		{
			for(int j = 0; j < win_size; j++)
			{					
				if(i == 0 || j == 0 || i == win_size - 1 || j == win_size - 1)
				{
					_descriptor.at<float>(k,a+0) = color.at<cv::Vec3b>(r+i,c+j)(0)/255.f;
					_descriptor.at<float>(k,a+1) = color.at<cv::Vec3b>(r+i,c+j)(1)/255.f;
					_descriptor.at<float>(k,a+2) = color.at<cv::Vec3b>(r+i,c+j)(2)/255.f;
					a+=3;
				}
			}
		}
	}

	/* average cost calculation */
	double end_tick = cv::getTickCount();
	_cost = end_tick - start_tick;
	if(keypts.size()){
		_cost /= keypts.size();
	}
}

/**
*	@brief Covert color space given an image.
*	@param img Image input.
*	@param color Output image in the desired color space.
*/
void ColorFeature::covert_color(const cv::Mat &img, cv::Mat &color){
	if(_color_space == CS_RGB){
		color = img.clone();
	}else if(_color_space == CS_HSV){
		cv::cvtColor(img, color, CV_BGR2HSV_FULL);
	}else if(_color_space == CS_LAB){
		cv::cvtColor(img, color, CV_BGR2Lab);
	}
}

/**
*	@brief Convert color space enum type to string.
*	@param cs Color space..
*/
std::string ColorFeature::cs2str(COLOR_SPACE cs){
	if(cs == CS_RGB) return "RGB";
	if(cs == CS_LAB) return "LAB";
	if(cs == CS_HSV) return "HSV";
	return "UNKNOWN";
}