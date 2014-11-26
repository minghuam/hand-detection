/* Simple tool for sampling hand images and masks */


#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/video/background_segm.hpp"
#include <cstdio>

#define GOOD_PATH(x) if(x[x.size() - 1] == '/') \
						x = x.substr(0, x.size() - 1)

void update(cv::BackgroundSubtractorMOG2 &MOG2, cv::Mat &img, cv::Mat &mask, bool learn_bg = false){
 
 	int erosion_size = 3;
 	int dilation_size = 1;

    // background subtraction
    cv::Mat fg_mask;
    cv::Mat bgr;
    cv::GaussianBlur(img, bgr, cv::Size(7,7), 1,1);
    MOG2(bgr, fg_mask, learn_bg ? -1 : 0);

    if(learn_bg){
    	return;
    }
    
    // threshold
    fg_mask = fg_mask > 240;
    
    // smooth
    cv::medianBlur(fg_mask, fg_mask, 5);
    
    // erosion
	cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                        cv::Size( 2*erosion_size+1, 2*erosion_size+1 ),
                                        cv::Point( erosion_size, erosion_size ) );
	cv::erode( fg_mask, fg_mask, element );
    
    element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                    cv::Point( dilation_size, dilation_size ) );
    
    cv::dilate(fg_mask, fg_mask, element);
    
    
    //more smoothing
	cv::medianBlur(fg_mask, fg_mask, 3);
	cv::GaussianBlur(fg_mask, fg_mask, cv::Size(3,3), 5);
	fg_mask = fg_mask > 240;

    // copy
    fg_mask.copyTo(mask);
}

int main(int argc, char **argv){
	if(argc < 2){
		std::cout << argv[0] << ": rgb_saving_dir [msk_saving_dir]" \
					<< std::endl;
		return 0;
	}

	std::string rgb_path;
	std::string msk_path;
	if(argc == 2){
		rgb_path = argv[1];
		GOOD_PATH(rgb_path);
		msk_path = rgb_path;
	}else{
		rgb_path = argv[1];
		msk_path = argv[2];
		GOOD_PATH(rgb_path);
		GOOD_PATH(msk_path);
	}

	std::cout << "rgb path: " << rgb_path << std::endl;
	std::cout << "msk path: " << msk_path << std::endl;


	cv::VideoCapture cap(0);
	if(!cap.isOpened()){
		std::cout << "Failed to open camera!" << std::endl;
		return -1;
	}

	cv::BackgroundSubtractorMOG2 MOG2;
	int learning_frames = 0;
	int max_learning_frames = 300;

	cv::Mat raw_img;
	cv::Mat msk_img;
	char buf[64];
	int index = 0;
	while(1){
		if(!cap.read(raw_img)){
			std::cout << "camera error!" << std::endl;
			break;
		}

		cv::imshow("raw", raw_img);

		//cv::imshow("hand", hd_img);

		bool is_learning = learning_frames < max_learning_frames;
		if(is_learning){
			learning_frames++;
		}

		update(MOG2, raw_img, msk_img, is_learning);

		if(!is_learning){
			cv::imshow("mask", msk_img);
		}

		int key = cv::waitKey(30) & 0xFF;

		if(key == 27){
			break;
		}else if(key == 's' && !is_learning){
			sprintf(buf, "%s/rgb_%04d.jpg", rgb_path.c_str(), index);
			std::cout << buf << std::endl;
			cv::imwrite(std::string(buf), raw_img);
			sprintf(buf, "%s/msk_%04d.jpg", msk_path.c_str(), index);
			std::cout << buf << std::endl;
			cv::imwrite(std::string(buf), msk_img);
			index++;
		}
	}

}