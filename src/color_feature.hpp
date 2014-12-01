/** @file color_feature.hpp
*	@brief ColorFeature class header.
*	@author minghuam
*/

#ifndef __COLOR_FEATURE_HPP
#define __COLOR_FEATURE_HPP

#include <iostream>
#include "feature.hpp"
#include "image_cache.hpp"

/* color space enum defines */
enum COLOR_SPACE{
	CS_RGB,
	CS_LAB,
	CS_HSV
};

/**
*	@brief RGB/HSV ColorFeature.
*/
class ColorFeature : public Feature{
public:
	
	ColorFeature();

	ColorFeature(COLOR_SPACE cs, int patch_size);

	std::string key() const;

	void compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts);

private:
	/* color space */
	COLOR_SPACE _color_space;
	
	/* patch size, must be an odd number */
	int _patch_size;

	/* feature size */
	int _dimension;

	static std::string cs2str(COLOR_SPACE cs);
	void covert_color(const cv::Mat &input, cv::Mat &output);
};

#endif /* __COLOR_FEATURE_HPP */