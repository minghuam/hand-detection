/** @file feature_extractor.hpp
*	@brief FeatureExtractor class header.
*	@author minghuam
*/

#ifndef __FEATURE_EXTRACTOR_HPP
#define __FEATURE_EXTRACTOR_HPP

/* -- Includes -- */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_cache.hpp"
#include "feature.hpp"

/** @brief Feature extractor class.
*/
class FeatureExtractor{
public:
	FeatureExtractor();

	~FeatureExtractor();

	int add_feature(Feature *feat);

	std::vector<Feature*> get_features();

	void compute(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypts);
	
	static void get_keypts(cv::Mat &img, std::vector<cv::KeyPoint> &keypts, int step_size);

private:
	std::vector<Feature*> _features;
	ImageCache *_cache;
};


#endif /* __FEATURE_EXTRACTOR_HPP */