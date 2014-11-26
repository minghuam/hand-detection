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
	ColorFeature(){
		ColorFeature(CS_RGB, 1);
	}

	ColorFeature(COLOR_SPACE cs, int patch_size, ImageCache *cache = NULL) {
		_color_space = cs;
		_patch_size = patch_size;
		_cache = cache;

		if(_patch_size == 1){
			_dimension =  3;
		}else{
			_dimension = _patch_size * 2 + (_patch_size - 2) * 2;
			_dimension *= 3;
		}
	}

	std::string key() const{
		return cs2str(_color_space);
	}

	void compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts, \
			cv::Mat &desc);
	
	int dimension() const{
		return _dimension;
	}

private:
	COLOR_SPACE _color_space;
	int _patch_size;
	int _dimension;

	std::string cs2str(COLOR_SPACE cs) const;
	void covert_color(const cv::Mat &input, cv::Mat &output);
};

#endif /* __COLOR_FEATURE_HPP */