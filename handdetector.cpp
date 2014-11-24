#include "handdetector.hpp"

HandDetector::HandDetector(){

}

HandDetector::~HandDetector(){

}

void HandDetector::train_and_save(std::vector<std::string> &rgb_files, std::vector<std::string> &mask_files, \
		std::string feature_set, std::string saving_dir){

	if(rgb_files.size() != mask_files.size()){
		std::cout << "Training RGB and MASK files are not equal!" << std::endl;
		return;
	}

    LcFeatureExtractor	extractor;
    LcRandomTreesR		classifier;
    extractor.set_extractor(feature_set);
    
    stringstream ss;

    for(int i=0;i<(int)rgb_files.size();i++)
    {
    	std::cout << rgb_files[i] << std::endl;
    	std::cout << mask_files[i] << std::endl;

        Mat color_img = imread(rgb_files[i]);
        Mat mask_img = imread(mask_files[i],0);
        
        imshow("color",color_img);
        imshow("mask",mask_img);
        waitKey(0);
        
        //////////////////////////////////////////
        //										//
        //		 EXTRACT/SAVE HISTOGRAM			//
        //										//
        //////////////////////////////////////////
        
        Mat hist;
        computeColorHist_HSV(color_img,hist);
        
        //////////////////////////////////////////
        //										//
        //		  TRAIN/SAVE CLASSIFIER			//
        //										//
        //////////////////////////////////////////
        
        Mat desc;
        Mat lab;
        vector<KeyPoint> kp;
        
        mask_img.convertTo(mask_img,CV_8UC1);
        extractor.work(color_img, desc, mask_img, lab,1, &kp);
        classifier.train(desc,lab);
        
        ss.str("");
        ss << saving_dir << "/model_" << i;
        cout << ss.str() << endl;
        classifier.save(ss.str());
    }
}

/* Copy old code */
void HandDetector::computeColorHist_HSV(Mat &src, Mat &hist)
{
	
	int bins[] = {4,4,4};
    if(src.channels()!=3) exit(1);
    
	//Mat tmp;
    //src.copyTo(tmp);
    
	Mat hsv;
    cvtColor(src,hsv,CV_BGR2HSV_FULL);
    
	int histSize[] = {bins[0], bins[1], bins[2]};
    Mat his;
    his.create(3, histSize, CV_32F);
    his = Scalar(0);   
    CV_Assert(hsv.type() == CV_8UC3);
    MatConstIterator_<Vec3b> it = hsv.begin<Vec3b>();
    MatConstIterator_<Vec3b> it_end = hsv.end<Vec3b>();
    for( ; it != it_end; ++it )
    {
        const Vec3b& pix = *it;
        his.at<float>(pix[0]*bins[0]/256, pix[1]*bins[1]/256,pix[2]*bins[2]/256) += 1.f;
    }
	
    // ==== Remove small values ==== //
    float minProb = 0.01;
    minProb *= hsv.rows*hsv.cols;
    Mat plane;
    const Mat *_his = &his;
	
    NAryMatIterator itt = NAryMatIterator(&_his, &plane, 1);   
    threshold(itt.planes[0], itt.planes[0], minProb, 0, THRESH_TOZERO);
    double s = sum(itt.planes[0])[0];
	
    // ==== Normalize (L1) ==== //
    s = 1./s * 255.;
    itt.planes[0] *= s;
    itt.planes[0].copyTo(hist);
}