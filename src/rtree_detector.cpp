/** @file rtree_detector.cpp
*	@brief RTreeDetector class implementation file.
*	@author minghuam
*/

#include "rtree_detector.hpp"

/**
*	@brief Constructor and parameter setup
*/
RTreeDetector::RTreeDetector(){
	_params.max_depth				= 10;
	_params.regression_accuracy		= 0.00f;
	_params.min_sample_count		= 10;

	_key = "RTree";
	_train_cost = 0.0;
	_detect_cost = 0.0;
}

/**
*	@brief Train a detector.
*	@param trainData Training data matrix, each row is a feature vector.
*	@param labels Ground truth for the training data, a n by 1 vector.
*/
void RTreeDetector::train(cv::Mat &trainData, cv::Mat &labels){

	double start_tick = cv::getTickCount();

	cv::Mat varType = cv::Mat::ones(trainData.cols + 1, 1, CV_8UC1) * CV_VAR_NUMERICAL;
	_random_tree.train(trainData, CV_ROW_SAMPLE, labels, cv::Mat(), cv::Mat(), varType, cv::Mat(), _params);

	double end_tick = cv::getTickCount();

	_train_cost = end_tick - start_tick;

	/* average time cost per training sample */
	if(trainData.rows){
		_train_cost /= trainData.rows;
	}
}

/**
*	@brief Predict hand response given features.
*	@param features Feature data matrix, each row is a feature vector.
*	@param responses Prediction output, a n by 1 vector.
*/
void RTreeDetector::predict(cv::Mat &features, cv::Mat &responses){

	double start_tick = cv::getTickCount();

	int n = features.rows;
	responses = cv::Mat::zeros( n, 1, 5);
	for(int i = 0; i < n; i++)
	{
		responses.at<float>(i,0) = _random_tree.predict(features.row(i));
	}

	double end_tick = cv::getTickCount();

	/* average time cost */
	_detect_cost = end_tick - start_tick;
	if(n){
		_detect_cost /= n;
	}
}