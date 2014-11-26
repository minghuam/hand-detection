#include <iostream>
#include <opencv2/opencv.hpp>
#include "handdetector.hpp"

#include <vector>

#include "color_feature.hpp"
#include "image_cache.hpp"

#include "tinydir.hpp"
#include "tinylog.hpp"


#define LIVE_VIDEO 1

int main(int argc, char **argv){

	ImageCache cache;
	ColorFeature rgb1(CS_RGB, 1, &cache);
	ColorFeature hsv1(CS_HSV, 1, &cache);
	ColorFeature hsv3(CS_HSV, 3, &cache);
	ColorFeature rgb3(CS_RGB, 3, &cache);

	cv::Mat img = cv::imread("test.jpg");

	std::vector<cv::KeyPoint> keypts;
	cache.img2keypts(img, keypts, 1);

	cv::Mat desc1 = cv::Mat::zeros(keypts.size(), rgb1.dimension(), CV_32F);
	cv::Mat desc3 = cv::Mat::zeros(keypts.size(), rgb3.dimension(), CV_32F);

	rgb1.compute(img, keypts, desc1);
	hsv1.compute(img, keypts, desc1);

	cache.print_status();

	rgb3.compute(img, keypts, desc3);
	hsv3.compute(img, keypts, desc3);

	cache.print_status();

	cache.clear();

	cache.print_status();
	/*
	HandDetector hd(10);

	std::string training_rgb_path = "./training/rgb";
	std::string training_mask_path = "./training/mask";
	std::string classifier_path = "./models/classifiers";
	std::string gfeat_path = "./models/histogram";
	
	std::vector<std::string> training_rgb_files = list_dir(training_rgb_path.c_str(), ".jpg");	
	std::vector<std::string> training_mask_files = list_dir(training_mask_path.c_str(), ".jpg");
	//hd.train_and_save(training_rgb_files, training_mask_files, "rvl", classifier_path, gfeat_path);

	std::vector<std::string> classifier_files = list_dir(classifier_path.c_str(), ".xml");
	std::vector<std::string> gfeat_files = list_dir(gfeat_path.c_str(), ".xml");
	hd.load_models("rvl", classifier_files, gfeat_files);

#ifdef LIVE_VIDEO
	cv::VideoCapture cap(0);
	if(!cap.isOpened()){
		std::cout << "Failed to open camera!" << std::endl;
		return -1;
	}
#else
	std::string testing_rgb_path = "./testing/rgb";
	std::string testing_mask_path = "./testing/mask";
	std::vector<std::string> testing_rgb_files = list_dir(testing_rgb_path.c_str(), ".jpg");	
	std::vector<std::string> testing_mask_files = list_dir(testing_mask_path.c_str(), ".jpg");
#endif

	cv::Mat raw_img;
	cv::Mat hd_img;
	int index = 0;
	while(1){

#ifdef LIVE_VIDEO
		if(!cap.read(raw_img)){
			std::cout << "camera error!" << std::endl;
			break;
		}
#else
		raw_img = cv::imread(testing_rgb_files[index]);
		index += 1;
		if(index > testing_rgb_files.size() - 1){
			index = 0;
		}
#endif

		cv::imshow("raw", raw_img);
		hd.test(raw_img, hd_img);

        cv::addWeighted(raw_img, 0.7, hd_img, 0.3, 0, hd_img);

		cv::imshow("hand", hd_img);

		int key = cv::waitKey(30) & 0xFF;
		if(key == 27){
			break;
		}
	}
	*/
}







