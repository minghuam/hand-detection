#ifndef __IMAGE_CACHE
#define __IMAGE_CACHE

#include <iostream>
#include <unordered_map>

#include "tinylog.hpp"
#include "cv_util.hpp"

class ImageCache{
public:
	void clear();
	void push(const std::string &key, const cv::Mat &img);
	int get(const std::string &key, cv::Mat &img);
	void print_status();

	void img2keypts(cv::Mat &img, std::vector<cv::KeyPoint> &keypts, int step_size);

private:
	std::unordered_map<std::string, cv::Mat> _images;
};

#endif /* __IMAGE_CACHE */