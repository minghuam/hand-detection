/** @file aggregate_feature.cpp
*	@brief AggregateFeature class implementation file.
*	@author minghuam
*/

#include "aggregate_feature.hpp"

/**
*	@brief Default constructor
*/
AggregateFeature::AggregateFeature(){
	_key = "UNKNOWN";
	_cost = 0.0;
}

/**
*	@brief Compute feature. NOT IMPLEMENTED!
*/
void AggregateFeature::compute(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypts){
}

/**
*	@brief Combine a list of features into one.
8	@param features A vector of feature pointers.
*/
void AggregateFeature::aggregate(std::vector<Feature*> features){

	double start_tick = cv::getTickCount();

	/* calculate maximum rows/cols */
	int rows = 0;
	int cols = 0;
	for(int i = 0; i < (int)features.size(); i++){
		cv::Mat m = features[i]->descriptor();
		if(rows < m.rows){
			rows = m.rows;
		}
		cols += m.cols;
	}

	/* allocate memory */
	_descriptor = cv::Mat::zeros(rows, cols, CV_32F);

	/* concatenate all features */
	int col = 0;
	_cost = 0;
	_key = "";
	for(int i = 0; i < (int)features.size(); i++){
		cv::Mat m = features[i]->descriptor();
		cv::Rect roi = cv::Rect(col, 0, m.cols, m.rows);
		col += m.cols;
		_descriptor(roi) = m;
		_cost += features[i]->cost();
		if(i){
			_key += "_";
		}
		_key += features[i]->key();
	}

	/* addup all feature calculation cost */
	double end_tick = cv::getTickCount();
	double aggregate_cost = end_tick - start_tick;
	if(features.size()){
		aggregate_cost /= features.size();
	}

	/* add aggreagation cost */
	_cost += aggregate_cost;
}