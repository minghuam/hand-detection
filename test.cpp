#include <iostream>
#include <opencv2/opencv.hpp>
#include "tinydir.hpp"
#include "handdetector.hpp"

#include <vector>

int main(int argc, char **argv){
	std::vector<std::string> mask_files = list_dir("./images/mask", ".jpg");
	std::vector<std::string> rgb_files = list_dir("./images/rgb", ".jpg");
	
	HandDetector hd;

	hd.train_and_save(rgb_files, mask_files, "rvl", "./models");
}







