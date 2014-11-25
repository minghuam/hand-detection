#include "handdetector.hpp"

HandDetector::HandDetector(int knn): _knn(knn){

}

HandDetector::~HandDetector(){

}

void HandDetector::load_models(std::string feature_set, std::vector<std::string> &model_files, std::vector<std::string> &gfeat_files){

    if(model_files.size() != gfeat_files.size()){
        std::cout << "Model and Global feature files are not equal!" << std::endl;
        return;
    }

    _extractor.set_extractor(feature_set);
    // clear data
    _global_feat = cv::Mat();
    _classifiers = std::vector<LcRandomTreesR>(model_files.size());

    for(int i = 0; i < model_files.size(); i++){
        FileStorage fs;
        fs.open(gfeat_files[i],FileStorage::READ);
        Mat hist;
        fs["hsv_histogram"] >> hist;
        fs.release();
        _global_feat.push_back(hist);

        _classifiers[i].load_full(model_files[i]);
    }

    flann::KMeansIndexParams indexParams;
    _flann  = flann::Index(_global_feat, indexParams);
}

void HandDetector::train_and_save(std::vector<std::string> &rgb_files, std::vector<std::string> &mask_files, \
		std::string feature_set, std::string model_saving_dir, std::string hist_saving_dir){

	if(rgb_files.size() != mask_files.size()){
		std::cout << "Training RGB and MASK files are not equal!" << std::endl;
		return;
	}

    _extractor.set_extractor(feature_set);

    // clear data
    _global_feat = cv::Mat();
    _classifiers = std::vector<LcRandomTreesR>(rgb_files.size());

    std::cout << "TRAINING: " << std::endl;    
    stringstream ss;

    for(int i = 0;i < rgb_files.size(); i++)
    {
    	std::cout << " " << rgb_files[i] << std::endl;
    	std::cout << " " << mask_files[i] << std::endl;

        Mat color_img = imread(rgb_files[i]);
        Mat mask_img = imread(mask_files[i],0);
        
        //imshow("color",color_img);
        //imshow("mask",mask_img);
        //waitKey(0);
        
        // Global Feature, we use HSV histogram
        Mat hist;
        computeColorHist_HSV(color_img,hist);

        _global_feat.push_back(hist);

        ss.str("");
        ss << hist_saving_dir << "/globalfeat_" << i << ".xml";
        FileStorage fs;
        fs.open(ss.str(),FileStorage::WRITE);
        fs << "hsv_histogram" << hist;
        fs.release();

        // Random Tree Classifier
        Mat desc;
        Mat lab;
        vector<KeyPoint> kp;
        
        //mask_img.convertTo(mask_img,CV_8UC1);
        _extractor.work(color_img, desc, mask_img, lab,1, &kp);
        
        LcRandomTreesR classifier;
        classifier.train(desc,lab);

        _classifiers[i] = classifier;
        
        ss.str("");
        ss << model_saving_dir << "/randtree_" << i;
        classifier.save(ss.str());
    }

    flann::KMeansIndexParams indexParams;
    _flann  = flann::Index(_global_feat, indexParams);

}


void HandDetector::test(Mat &img, Mat &dsp, int num_models, float color_code)
{
    if(num_models>_knn) return;
    
    Mat hist;
    computeColorHist_HSV(img,hist);                                 // extract hist
    
    std::vector<int> indices;
    std::vector<float> dists;
    _flann.knnSearch(hist, indices, dists, _knn);            // probe search
    
    Mat descriptors;
    vector<KeyPoint> kp;

    _extractor.work(img, descriptors, 3, &kp);
    
    Mat response_avg = Mat::zeros(descriptors.rows, 1, CV_32FC1); 
    Mat response_vec;

    float norm = 0;
    for(int i=0;i<num_models;i++)
    {
        int idx = indices[i];
        _classifiers[idx].predict(descriptors, response_vec);       // run classifier
        
        response_avg += response_vec*float(pow(0.9f,(float)i));
        norm += float(pow(0.9f,(float)i));
    }
    
    response_avg /= norm;
    
    cv::Size sz = img.size();
    int bs = _extractor.bound_setting; 
    Mat response_img;
    rasterizeResVec(response_img, response_avg, kp, sz, bs);
    colormap(response_img, _raw, 1, color_code);

    vector<Point2f> pt;
    _ppr = postprocess(response_img,pt);
    
    colormap(_ppr,_ppr,1, color_code);
    
    Mat _dsp;

    if(0)
    {
        
        imshow("hd response",_dsp);
        imshow("hd img",img);   
        imshow("_ppr",_ppr);    
        waitKey(1);
    }
    
    //dsp = pp;
    if(!dsp.data) dsp = _dsp;                   // pass by reference?
    
}


Mat HandDetector::postprocess(Mat &img,vector<Point2f> &pt)
{
    Mat tmp;
    //GaussianBlur(img,tmp,cv::Size(31,31),0,0,BORDER_REFLECT);
    GaussianBlur(img,tmp,cv::Size(15,15),0,0,BORDER_REFLECT);
    
    //Mat dsp;
    colormap(tmp, _blu, 1);
    //imshow("dsp",dsp);
    
    tmp = tmp > 0.04;
    
    vector<vector<cv::Point> > co;
    vector<Vec4i> hi;
    
    //findContours(tmp,co,hi,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(tmp,co,hi,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
    tmp *= 0;
    
    float max_size = 0;
    for(int i=0;i<(int)co.size();i++)
    {
        float area = contourArea(Mat(co[i]));
        if(area < 300) continue;
        if( area > max_size)
        {
            tmp *= 0;
            drawContours(tmp, co,i, CV_RGB(255,255,255), CV_FILLED, CV_AA);
            max_size = contourArea(Mat(co[i]));
        }
    }
    
    return tmp;

}


void HandDetector::rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, cv::Size s, int bs)
{   
    if((img.rows!=s.height) || (img.cols!=s.width) || (img.type()!=CV_32FC1) ) img = Mat::zeros( s, CV_32FC1);
    
    for(int i = 0;i< (int)keypts.size();i++)
    {
        int r = floor(keypts[i].pt.y);
        int c = floor(keypts[i].pt.x);
        img.at<float>(r,c) = res.at<float>(i,0);
    }
}


void HandDetector::colormap(Mat &src, Mat &dst, int do_norm, float color_code)
{
    
    double minVal,maxVal;
    minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
    
    //cout << "colormap minmax: " << minVal << " " << maxVal << " Type:" <<  src.type() << endl;
    
    Mat im;
    src.copyTo(im);
    
    if(do_norm) im = (src-minVal)/(maxVal-minVal);      // normalization [0 to 1]
    
    Mat mask;   
    mask = Mat::ones(im.size(),CV_8UC1)*255.0;  
    
    compare(im,0.01,mask,CMP_GT);                       // one color values greater than X  
    
    
    Mat U8;
    im.convertTo(U8,CV_8UC1,255,0);
    
    Mat I3[3],hsv;
    I3[0] = U8 * color_code;
    I3[1] = mask;
    I3[2] = mask;
    merge(I3,3,hsv);
    cvtColor(hsv,dst,CV_HSV2RGB_FULL);
    
    
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