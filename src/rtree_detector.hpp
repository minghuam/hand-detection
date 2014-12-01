/** @file rtree_detector.hpp
*	@brief RTreeDetector class header file.
*	@author minghuam
*/

#ifndef __RTREE_DETECTORH_HPP
#define __RTREE_DETECTORH_HPP

#include <opencv2/opencv.hpp>
#include "detector.hpp"

/**
*	@brief Pixel wise hand detection with random forest regressor.
*/
class RTreeDetector : public Detector{
public:
	RTreeDetector();

	void train(cv::Mat &trainData, cv::Mat &labels);
	
	void predict(cv::Mat &features, cv::Mat &responses);

private:
	CvRTParams _params;
	CvRTrees _random_tree;
};

#endif /* __RTREE_DETECTORH_HPP */