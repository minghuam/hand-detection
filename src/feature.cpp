#include "feature.hpp"
#include "tinylog.hpp"

/**
*	@brief Getter for key.
*	@return key string.
*/
std::string Feature::key() const{
	return _key;
}

/**
*	@brief Getter for cost.
*	@return Average cost.
*/
double Feature::cost() const{
	return _cost;
}

/**
*	@brief Getter for feature descriptor.
*	@return Descriptor.
*/
cv::Mat Feature::descriptor() const{
	return _descriptor;
}

/**
*	@brief Enable cache.
*/
void Feature::enable_cache(ImageCache *cache){
	_cache = cache;
}

/**
*	@brief print feature summary.
*/
void Feature::print() const{
	LOGF("FEATURE: key: %s, rows: %d, cols: %d, cost: %.8f",\
		_key.c_str(), _descriptor.rows, _descriptor.cols, _cost);
}
