#include "feature_extractor.hpp"

/**
*	@brief Constructor
*/
FeatureExtractor::FeatureExtractor(){
	_cache = new ImageCache();
}

/**
*	@brief Destructor
*	Clear cache, delete features
*/
FeatureExtractor::~FeatureExtractor(){
	delete _cache;
	_cache = NULL;

	for(int i = 0; i < _features.size(); i++){
		if(_features[i] != NULL){
			delete _features[i];
			_features[i] = NULL;
		}
	}
}

/**
*	@brief Get all features
*	@return A vector of feature pointers.
*/
std::vector<Feature*> FeatureExtractor::get_features(){
	return _features;
}

/**
*	@brief Add one feature
*/
int FeatureExtractor::add_feature(Feature *feat){
	
	//feat->enable_cache(_cache);
	
	_features.push_back(feat);

	return 0;
}

/**
*	@brief Compute features given an image and key points.
*	@param img Input image
*	@param keypts Key points to compute features.
*/
void FeatureExtractor::compute(const cv::Mat &img, \
	const std::vector<cv::KeyPoint> &keypts){
	
	if(_features.size() == 0){
		return;
	}

	if(_cache){
		_cache->clear();
	}

	for(int i = 0; i < _features.size(); i++){
		_features[i]->compute(img, keypts);
	}
}

/**
*	@brief Compute key points given an image and step size
*	@param img Input image
*	@param keypts Key points output.
*	@param step_size Step size between pixels
*/
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