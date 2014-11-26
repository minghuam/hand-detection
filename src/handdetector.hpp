#ifndef _HANDDETECTOR_HPP
#define _HANDDETECTOR_HPP

#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>
#include "HandDetector/FeatureComputer.hpp"
#include "HandDetector/Classifier.h"
#include "HandDetector/LcBasic.h"

class HandDetector
{
private:
	std::vector<LcRandomTreesR* > _classifiers;
	LcFeatureExtractor *_extractor;
	flann::Index *_flann;
	cv::Mat	*_global_feat;

	int _knn;

public:
	HandDetector(int knn);
	~HandDetector();

	void reset();

	void train_and_save(std::vector<std::string> &rgb_files, std::vector<std::string> &mask_files, \
		std::string feature_set, std::string model_saving_dir, std::string feat_saving_dir);

	void load_models(std::string feature_set, std::vector<std::string> &classifier_files, std::vector<std::string> &gfeat_files);

	void test(Mat &img, Mat &dsp, float color_code = 0.85);
	Mat postprocess(Mat &img,vector<Point2f> &pt);
	void rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, cv::Size s);
	void colormap(Mat &src, Mat &dst, int do_norm, float color_code = 0.85);

	void computeColorHist_HSV(cv::Mat &src, cv::Mat &hist);

	Mat							_raw;				// raw response
	Mat							_blu;				// blurred image
	Mat							_ppr;				// post processed

	
};

#endif /* End of HandDetector */