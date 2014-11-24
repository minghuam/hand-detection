#ifndef _HANDDETECTOR_HPP
#define _HANDDETECTOR_HPP

#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "HandDetector/FeatureComputer.hpp"
#include "HandDetector/Classifier.h"

class HandDetector
{
public:
	HandDetector();
	~HandDetector();

	void train_and_save(std::vector<std::string> &rgb_files, std::vector<std::string> &mask_files, \
		std::string feature_set, std::string saving_dir);

	void computeColorHist_HSV(cv::Mat &src, cv::Mat &hist);

};

#endif /* End of HandDetector */