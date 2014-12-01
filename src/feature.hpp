/** @file feature.hpp
*	@brief Feature class header file.
*	@author minghuam
*/

#ifndef __FEATURE_HPP
#define __FEATURE_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "image_cache.hpp"

/**
*	@brief Feature abstract class.
*/
class Feature{
public:

	virtual void compute(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypts) = 0;

	std::string key() const;
	
	double cost() const;

	cv::Mat descriptor() const;

	void enable_cache(ImageCache *cache);

	void print() const;

protected:
	/* cache for intermediate results */
	ImageCache *_cache;

	/* unique string key for this feature */
	std::string _key;

	/* feature descriptor */
	cv::Mat _descriptor;

	/* average cost per key point to calculate this feature */
	double _cost;
};

#endif /* __FEATURE_HPP */