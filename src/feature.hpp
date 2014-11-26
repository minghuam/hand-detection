#ifndef __FEATURE_HPP
#define __FEATURE_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "image_cache.hpp"

class Feature{
public:
	virtual void compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts, \
			cv::Mat &desc) = 0;

	virtual std::string key() const = 0;
	virtual int dimension() const = 0;
	
protected:
	ImageCache *_cache;
};

#endif /* __FEATURE_HPP */