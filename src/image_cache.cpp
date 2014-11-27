#include "image_cache.hpp"

void ImageCache::clear(){
	_images.clear();
}

void ImageCache::push(const std::string &key, const cv::Mat &img){
	_images[key] = img.clone();
}

int ImageCache::get(const std::string &key, cv::Mat &img){
	if(_images.find(key) != _images.end()){
		img = _images[key].clone();
		LOG("cache hit");
		return 0;
	}else{
		LOG("cache miss");
		return -1;
	}
}

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