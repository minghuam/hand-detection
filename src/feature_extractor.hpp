/** @file feature_extractor.hpp
*	@brief Feature extractors.
*	@author minghuam
*/

#ifndef __FEATURE_EXTRACTOR_HPP
#define __FEATURE_EXTRACTOR_HPP

/* -- Includes -- */
#include <iostream>


/** @brief Feature extractor abstract class
*/
class FeatureExtractor{

public:
	virtual void load(std::string &file) = 0;
	virtual void save(std::string &file) = 0;
	virtual void compute(cv::Mat &img, cv::Mat &desc);


};


#endif /* __FEATURE_EXTRACTOR_HPP */