#include "feature_extractor.hpp"

FeatureExtractor::FeatureExtractor(){
	_dimension = 0;
	LOG("cache created!");
	_cache = new ImageCache();
}

FeatureExtractor::~FeatureExtractor(){
	delete _cache;
	_cache = NULL;
}

int FeatureExtractor::add_feature(Feature *feat){
	feat->enable_cache(_cache);
	_features.push_back(feat);
	_dimension += feat->dimension();
	return 0;
}

void FeatureExtractor::compute(const cv::Mat &img, \
	const std::vector<cv::KeyPoint> keypts, cv::Mat &desc){
	
	if(_features.size() == 0){
		return;
	}

	LOG("compute");

	if(_cache){
		_cache->clear();
	}
	
	LOG("compute");

	if(desc.rows != keypts.size() || desc.cols != _dimension){
		desc = cv::Mat::zeros(keypts.size(), _dimension, CV_32F);
	}

	int d = 0;
	for(int i = 0; i < _features.size(); i++){
		cv::Mat roi = desc(cv::Rect(d, 0, _features[i]->dimension(), keypts.size()));
		_features[i]->compute(img, keypts, roi);
		d += _features[i]->dimension();
	}

}

void FeatureExtractor::get_keypts(cv::Mat &img, std::vector<cv::KeyPoint> &keypts, int step_size){
	cv::DenseFeatureDetector dfd;
	float	initFeatureScale	= 1.f;				// inital size
	int		featureScaleLevels	= 1;				// one level
	float	featureScaleMul		= 1.00f;			// multiplier (ignored if only one level)
	int		train_initXyStep	= step_size;		// space between pixels for training (must be 1)
	dfd = cv::DenseFeatureDetector(initFeatureScale,featureScaleLevels,\
		featureScaleMul,train_initXyStep);
	dfd.detect(img,keypts);
}