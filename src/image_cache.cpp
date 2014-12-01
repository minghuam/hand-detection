/** @file image_cache.cpp
*	@brief ImageCache class implementation file.
*	@author minghuam
*/

#include "image_cache.hpp"

/**
*	@brief Clear the cache.
*/
void ImageCache::clear(){
	_images.clear();
}

/**
*	@brief Push a image into the cache.
*	@param key String key to access this image.
*	@param img Image to push.
*/
void ImageCache::push(const std::string &key, const cv::Mat &img){
	/* save images in a hash map */
	_images[key] = img.clone();
}

/**
*	@brief Given a string key, fetch the image.
*	@param key String key to access this image.
*	@param img Output image.
*	@return 0 for success, -1 for failure.
*/
int ImageCache::get(const std::string &key, cv::Mat &img){
	if(_images.find(key) != _images.end()){
		img = _images[key].clone();
		return 0;
	}else{
		return -1;
	}
}

/**
*	@brief print the current status of the cache.
*/
void ImageCache::print_status(){
	if(_images.size() == 0){
		LOG("Image cache is empty.");
		return;
	}
	LOG("Image cache status:");
	std::unordered_map<std::string, cv::Mat>::iterator it= _images.begin();
	while(it != _images.end()){
		cv::Mat m = it->second;
		std::string key = it->first;
		LOGF("	%s: (%d, %d, %s)", \
			key.c_str(), m.rows, m.cols, CV_TYPE2STR(m.type()).c_str());
		++it;
	}
}