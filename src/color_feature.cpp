#include "color_feature.hpp"

ColorFeature::ColorFeature(){
	ColorFeature(CS_RGB, 1);
}

ColorFeature::ColorFeature(COLOR_SPACE cs, int patch_size) {
	_color_space = cs;
	_patch_size = patch_size;

	if(_patch_size == 1){
		_dimension =  3;
	}else{
		_dimension = _patch_size * 2 + (_patch_size - 2) * 2;
		_dimension *= 3;
	}
}

std::string ColorFeature::key() const{
	return cs2str(_color_space);
}

void ColorFeature::compute(const cv::Mat &img, \
			const std::vector<cv::KeyPoint> &keypts, \
			cv::Mat &desc){
	
	cv::Mat color;
	if(_cache != NULL){
		if(_cache->get(key(), color) == -1){
			covert_color(img, color);
			_cache->push(key(), color);
		}
	}else{
		covert_color(img, color);
	}

	int win_size = _patch_size;
	for(int k = 0; k < (int)keypts.size(); k++)
	{
		int r = int(floor(.5 + keypts[k].pt.y) - floor(win_size*0.5));
		int c = int(floor(.5 + keypts[k].pt.x) - floor(win_size*0.5));
		int a = 0;
		for(int i = 0; i < win_size; i++)
		{
			for(int j = 0; j < win_size; j++)
			{					
				if(i == 0 || j == 0 || i == win_size - 1 || j == win_size - 1)
				{
					desc.at<float>(k,a+0) = color.at<cv::Vec3b>(r+i,c+j)(0)/255.f;
					desc.at<float>(k,a+1) = color.at<cv::Vec3b>(r+i,c+j)(1)/255.f;
					desc.at<float>(k,a+2) = color.at<cv::Vec3b>(r+i,c+j)(2)/255.f;
					a+=3;
				}
			}
		}
	}
}

void ColorFeature::covert_color(const cv::Mat &img, cv::Mat &color){
	if(_color_space == CS_RGB){
		color = img.clone();
	}else if(_color_space == CS_HSV){
		cv::cvtColor(img, color, CV_BGR2HSV_FULL);
	}else if(_color_space == CS_LAB){
		cv::cvtColor(img, color, CV_BGR2Lab);
	}
}

std::string ColorFeature::cs2str(COLOR_SPACE cs) const{
	if(cs == CS_RGB) return "RGB";
	if(cs == CS_LAB) return "LAB";
	if(cs == CS_HSV) return "HSV";
	return "UNKNOWN";
}