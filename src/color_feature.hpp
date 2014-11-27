#ifndef __COLOR_FEATURE_HPP
#define __COLOR_FEATURE_HPP

#include <iostream>
#include "feature.hpp"
#include "image_cache.hpp"

enum COLOR_SPACE{
	CS_RGB,
	CS_LAB,
	CS_HSV
};

class ColorFeature : public Feature{
public:
	ColorFeature();
	ColorFeature(COLOR_SPACE cs, int patch_size);

	std::string key() const;

	void compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts, \
			cv::Mat &desc);

private:
	COLOR_SPACE _color_space;
	int _patch_size;

	std::string cs2str(COLOR_SPACE cs) const;
	void covert_color(const cv::Mat &input, cv::Mat &output);
};

#endif /* __COLOR_FEATURE_HPP */