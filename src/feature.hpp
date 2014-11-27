#ifndef __FEATURE_HPP
#define __FEATURE_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "image_cache.hpp"

class Feature{
public:
	Feature();
	
	virtual void compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts, \
			cv::Mat &desc) = 0;

	virtual std::string key() const = 0;
	
	int dimension() const;
	void enable_cache(ImageCache *cache);
	
protected:
	ImageCache *_cache;
	int _dimension;
};

#endif /* __FEATURE_HPP */