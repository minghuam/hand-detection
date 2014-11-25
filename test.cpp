#include <iostream>
#include <opencv2/opencv.hpp>
#include "tinydir.hpp"
#include "handdetector.hpp"

#include <vector>

int main(int argc, char **argv){

	HandDetector hd(3);

	std::string mask_path = "./images/mask";
	std::string rgb_path = "./images/rgb";
	std::string classifier_path = "./models/classifiers";
	std::string gfeat_path = "./models/histogram";
	
	hd.train_and_save(rgb_files, mask_files, "rvl", classifier_path, gfeat_path);

	//std::vector<std::string> mask_files = list_dir(mask_path, ".jpg");
	//std::vector<std::string> rgb_files = list_dir(rgb_path, ".jpg");
	
	//hd.load_models(model_files, gfeat_files);



}







