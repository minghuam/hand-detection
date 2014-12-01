/** @file aggregate_feature.hpp
*	@brief AggregateFeature class header.
*	@author minghuam
*/

#ifndef __AGGREGATE_FEATURE_HPP
#define __AGGREGATE_FEATURE_HPP

#include <opencv2/opencv.hpp>
#include "feature.hpp"
#include "feature_extractor.hpp"

/**
*	@brief AggregateFeature from other features.
*/
class AggregateFeature : public Feature{

public:
	AggregateFeature();

	void compute(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypts);

	void aggregate(std::vector<Feature*> features);
};

#endif