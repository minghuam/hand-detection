/** @file image_cache.hpp
*	@brief ImageCache class header file.
*	@author minghuam
*/

#ifndef __IMAGE_CACHE
#define __IMAGE_CACHE

#include <iostream>
#include <unordered_map>

#include "tinylog.hpp"
#include "cv_util.hpp"

/**
*	@brief Cache for intermediate results.
*	This is useful especially for saving preprocessing results.
*/
class ImageCache{
public:
	void clear();
	void push(const std::string &key, const cv::Mat &img);
	int get(const std::string &key, cv::Mat &img);
	void print_status();
private:
	std::unordered_map<std::string, cv::Mat> _images;
};

#endif /* __IMAGE_CACHE */