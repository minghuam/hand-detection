/** @file detector.hpp
*	@brief Detector class header.
*	@author minghuam
*/

#ifndef __DETECTOR_HPP
#define __DETECTOR_HPP

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

/**
*	@brief Detector abstract class.
*/
class Detector{
public:
	virtual void train(cv::Mat &trainData, cv::Mat &labels) = 0;
	virtual void predict(cv::Mat &features, cv::Mat &responses) = 0;
	std::string key() const;
	double train_cost() const;
	double detect_cost() const;
	void print() const;

protected:
	double _train_cost;
	double _detect_cost;
	std::string _key;
};

#endif