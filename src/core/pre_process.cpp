#include "easyocr/core/pre_process.hpp"

using namespace std;
using namespace cv;



TextDetector::TextDetector(){
//    Detectorparams = params;
//    imageDirectory = image_dir;
    
}


cv::Mat TextDetector::preProcess(cv::Mat &image){
    cv::Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    return gray;
}



void TextDetector::findKEdgeFirst(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols){
    int count = 0;
    for (int i = 0; i < data.cols; i ++) {
        uchar *u = data.ptr<uchar>(i);
        for (int j = 0; j < data.rows; j ++) {
            if(edgeValue == (int)u[j]){
                if(count < k){
                    count ++;
                    cols.push_back(i);
                    rows.push_back(j);
                }

            }

        }
    }

}

void TextDetector::findKEdgeLast(cv::Mat &data, int edgeValue,int k,vector<int> &rows, vector<int> &cols){
    int count = 0;
    for (int i = data.cols - 1; i >= 0; i --) {
        uchar *u = data.ptr<uchar>(i);
        for (int j = data.rows - 1; j >= 0; j --) {
            if(edgeValue == (int)u[j]){
                if(count < k){
                    count ++;
                    cols.push_back(i);
                    rows.push_back(j);
                }

            }
        }

    }

}

void TextDetector::adaptiveHistEqual(cv::Mat &src,cv::Mat &dst,double clipLimit)
{
    Ptr<cv::CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->apply(src, dst);
}


void TextDetector::sharpenImage(const cv::Mat &image, cv::Mat &result)
{
    //创建并初始化滤波模板
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    kernel.at<float>(2,1) = -1.0;

    result.create(image.size(),image.type());

    //对图像进行滤波
    cv::filter2D(image,result,image.depth(),kernel);
}


bool TextDetector::verifyCharSizes(Mat r) {
  // Char sizes 45x90
  float aspect = 45.0f / 90.0f;
  float charAspect = (float)r.cols / (float)r.rows;
  float error = 0.7f;
  float minHeight = 10.f;
  float maxHeight = 35.f;
  // We have a different aspect ratio for number 1, and it can be ~0.2
  float minAspect = 0.05f;
  float maxAspect = aspect + aspect * error;
  // area of pixels
  int area = cv::countNonZero(r);
  // bb area
  int bbArea = r.cols * r.rows;
  //% of pixel in area
  int percPixels = area / bbArea;

  if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
      r.rows >= minHeight && r.rows < maxHeight)
    return true;
  else
    return false;
}



int TextDetector::sobelOper(const Mat &in, Mat &out, int blurSize)
{
  Mat mat_blur;
  mat_blur = in.clone();
  GaussianBlur(in, mat_blur, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);

  Mat mat_gray;
  if (mat_blur.channels() == 3)
    cvtColor(mat_blur, mat_gray, CV_RGB2GRAY);
  else
    mat_gray = mat_blur;

  int scale = SOBEL_SCALE;
  int delta = SOBEL_DELTA;
  int ddepth = SOBEL_DDEPTH;

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;


  Sobel(mat_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  Sobel(mat_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);

  Mat grad;
  addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_X_WEIGHT, 0, grad);

  out = grad;

//  Mat mat_threshold;
//  double otsu_thresh_val =
//      threshold(grad, mat_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);


//  Mat element = getStructuringElement(MORPH_RECT, Size(morphW, morphH));
//  morphologyEx(mat_threshold, mat_threshold, MORPH_CLOSE, element);

//  out = mat_threshold;


  return 0;
}


Mat TextDetector::preprocessChar(Mat in) {
  // Remap image
  int h = in.rows;
  int w = in.cols;

  int charSize = CHAR_SIZE;

  Mat transformMat = Mat::eye(2, 3, CV_32F);
  int m = max(w, h);
  transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
  transformMat.at<float>(1, 2) = float(m / 2 - h / 2);

  Mat warpImage(m, m, in.type());
  warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
             BORDER_CONSTANT, Scalar(0));

  Mat out;
  resize(warpImage, out, Size(charSize, charSize));

  return out;
}

void TextDetector::setMorParameters(int char_size)
{
    switch(char_size)
    {
        case BIGCHAR:
           src_open_val = Size(5, 5);
           src_dilate_val = Size(10, 20);
           src_erode_val = Size(5, 5);
        break;
        case MEDCHAR:
            src_open_val = Size(5, 5);
            src_dilate_val = Size(5, 13);
            src_erode_val = Size(5, 5);
        break;
        case SMALLCHAR:
            src_open_val = Size(3, 5);
            src_dilate_val = Size(1, 8);
            src_erode_val = Size(1, 5);
        break;
        default:
           std::cout<<"char size input is wrong!!! use default big char parameters."<<std::endl;
           src_open_val = Size(5, 5);
           src_dilate_val = Size(5, 20);
           src_erode_val = Size(5, 5);
        break;
    }
}

void TextDetector::setThreParameters(int char_color)
{
    switch(char_color)
    {
        case WHITE:
           inv_bin = false;
        break;
        case BLACK:
           inv_bin = true;
        break;
        default:
           std::cout<<"char color input is wrong!!! Use default white char set."<<std::endl;
           inv_bin = false;
        break;
    }
}

int TextDetector::slidingWnd(Mat& src, vector<Mat>& wnd, Size wndSize, double x_percent, double y_percent)
{
    std::cout<<"size = "<<wndSize<<std::endl;
    int count = 0;  //记录滑动窗口的数目
    int x_step = cvCeil(x_percent*wndSize.width);
    int y_step = cvCeil(y_percent*wndSize.height);
//    int x_step = 1;
//    int y_step = 1;
    int64 count1 = getTickCount();
    double freq = getTickFrequency();
    std::cout<<"picece_orc size is "<<src.rows<<" * "<<src.cols<<std::endl;

    //利用窗口对图像进行遍历
    for (int i = 0; i < src.rows- wndSize.height; i+=y_step)
    {
        for (int j = 0; j < src.cols- wndSize.width; j+=x_step)
        {
            Rect roi(Point(j, i), wndSize);
            Mat ROI = src(roi);          
            cv::Mat idx;
            findNonZero(ROI, idx);
            int one_count = (int)idx.total();

            int zero_count = (int)ROI.total() - one_count;
            float one_percent = (float)one_count/(float)(one_count+zero_count);
//            std::cout<<"one_count = "<<one_count<<std::endl;
//            std::cout<<"zero_count = "<<zero_count<<std::endl;
//            std::cout<<"one_percent = "<<one_percent<<std::endl;
            if(one_percent>0.1)
            {
                 wnd.push_back(ROI);
                 count++;
            }
        }
    }

    int64 count2 = getTickCount();
    double time = (count2 - count1) / freq;
    cout << "slide Time=" << time * 100 << "ms"<<endl;
    return count;
}


//segment the spine text
void TextDetector::segmentSrcSlide(cv::Mat &spineGray, vector<Mat> &single_char_vec, int im_num, bool save)
{
    srand((unsigned)time(NULL));
    ////set parameters
//    int char_size;
//#ifdef DEBUG
//    printf("please input char size: big is 1, mediate is 2, small is 3\n");
//    scanf("%d",&char_size);
//#endif
//    setMorParameters(char_size);
//    int char_color;
//#ifdef DEBUG
//    printf("please input char size: white is 1, black is 2\n");
//    scanf(" %d",&char_color);
//#endif
//    setThreParameters(char_color);
//#ifdef DEBUG
//    std::cout<<"char_size = "<<char_size<<","<<"char_color = "<<char_color<<std::endl;
//#endif


    ////gauss smoothing
    int m_GaussianBlurSize = 5;
    Mat mat_blur;
    GaussianBlur(spineGray, mat_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);
#ifdef DEBUG
    while(1)
    {
      imshow("src_gauss", mat_blur);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////histequal and sharpen
    Mat spineGrayTemp = mat_blur - 0.5;
    cv::Mat spineAhe;
    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);
    cv::Mat spineShrpen;
    sharpenImage(spineAhe, spineShrpen);
#ifdef DEBUG
    while(1)
    {
      imshow("sharpen", spineShrpen);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////threshold
    cv::Mat thresh_src;
    if(inv_bin)
    {
        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY_INV);
    }
    else
    {
        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
    }
#ifdef DEBUG
    while(1)
    {
      imshow("thresh_src", thresh_src);
      if(char(cvWaitKey(15))==27)break;
    }
#endif


    ////slide window in src
    vector<Mat> charWnd;
    int count=slidingWnd(thresh_src, charWnd, Size(100, 130),0.1,0.1);
    std::cout<<"slide count is "<<count<<std::endl;


    ////save single char image after segment
    for(int char_num=0;char_num<charWnd.size();char_num++)
    {
         Mat single_char_=charWnd.at(char_num);
         Mat single_char;
         single_char = preprocessChar(single_char_);
         single_char_vec.push_back(single_char);
        if(save)
        {
            const char* single_char_folder_ = "../../../src/easyocr/char_img";
            std::stringstream ss(std::stringstream::in | std::stringstream::out);
            ss << single_char_folder_ << "/" << im_num << "_src" << char_num/*<<"_"<<rand()*/<< ".jpg";
            imwrite(ss.str(),single_char);
        }
#ifdef DEBUG
        while(1)
        {
          imshow( "single_char", single_char_ );
          if(char(cvWaitKey(15))==27)break;
        }
#endif
    }

#ifdef DEBUG
    cvDestroyWindow("sharpen");
    cvDestroyWindow("src_gauss");
    cvDestroyWindow("thresh_src");
    cvDestroyWindow("single_char");
#endif
}



//segment the spine text
void TextDetector::segmentSrcMor(cv::Mat &spineGray, vector<Mat> &single_char_vec, int im_num, bool save)
{
    srand((unsigned)time(NULL));
    ////set parameters
    int char_size;
#ifdef DEBUG
    printf("please input char size: big is 1, mediate is 2, small is 3\n");
    scanf("%d",&char_size);
#endif
    setMorParameters(char_size);
    int char_color;
#ifdef DEBUG
    printf("please input char size: white is 1, black is 2\n");
    scanf(" %d",&char_color);
#endif
    setThreParameters(char_color);
#ifdef DEBUG
    std::cout<<"char_size = "<<char_size<<","<<"char_color = "<<char_color<<std::endl;
#endif


    ////gauss smoothing
    int m_GaussianBlurSize = 5;
    Mat mat_blur;
    GaussianBlur(spineGray, mat_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);
#ifdef DEBUG
    while(1)
    {
      imshow("src_gauss", mat_blur);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////histequal and sharpen
    Mat spineGrayTemp = mat_blur - 0.5;
    cv::Mat spineAhe;
    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);
    cv::Mat spineShrpen;
    sharpenImage(spineAhe, spineShrpen);
#ifdef DEBUG
    while(1)
    {
      imshow("sharpen", spineShrpen);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////threshold
    cv::Mat thresh_src;
    if(inv_bin)
    {
        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY_INV);
    }
    else
    {
        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
    }



    ////morphological open
    Mat element_src = getStructuringElement(MORPH_RECT, src_open_val);
    Mat open_src;
    morphologyEx(thresh_src,open_src,MORPH_OPEN,element_src);
#ifdef DEBUG
    while(1)
    {
      imshow("open_src", open_src);
      if(char(cvWaitKey(15))==27)break;
    }
#endif
    cv::Mat thres_window = open_src.clone();


    //////morphological close
    Mat elementDilate = getStructuringElement(MORPH_RECT, src_dilate_val);
    Mat elementErode = getStructuringElement(MORPH_RECT, src_erode_val);
    Mat dilate_out,erode_out;
    morphologyEx(thres_window,dilate_out,MORPH_DILATE,elementDilate);
    morphologyEx(dilate_out,erode_out,MORPH_ERODE,elementErode);
//    erode_out = dilate_out.clone();
#ifdef DEBUG
    while(1)
    {
      imshow("erode_out", erode_out);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////find contours
    Mat img_contours;
    erode_out.copyTo(img_contours);
    vector<vector<Point> > contours;
    findContours(img_contours,
                 contours,               // a vector of contours
                 CV_RETR_EXTERNAL,       // retrieve the external contours
                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
    Mat sepertate_im(thres_window.size(),thres_window.depth(),Scalar(255));
    drawContours(sepertate_im,contours,-1,Scalar(0),2);
#ifdef DEBUG
    while(1)
    {
      imshow("sepertate_im",sepertate_im);
      if(char(cvWaitKey(15))==27)break;
    }
#endif


    ////remove noise
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<Rect> vecRect;
    while (itc != contours.end())
    {
      Rect mr = boundingRect(Mat(*itc));
      Mat auxRoi(thres_window, mr);
      if (/*verifyCharSizes(auxRoi)*/1) vecRect.push_back(mr);
      ++itc;
    }

    ////save single char image after segment
    for(int char_num=0;char_num<vecRect.size();char_num++)
    {
         Mat single_char_=thres_window(vecRect.at(char_num));
         Mat single_char;
         single_char = preprocessChar(single_char_);
         single_char_vec.push_back(single_char);
        if(save)
        {
            const char* single_char_folder_ = "../../../src/easyocr/char_img";
            std::stringstream ss(std::stringstream::in | std::stringstream::out);
            ss << single_char_folder_ << "/" << im_num << "_src" << char_num<<"_"<<rand()<< ".jpg";
            imwrite(ss.str(),single_char);
        }
#ifdef DEBUG
        while(1)
        {
          imshow( "single_char", single_char_ );
          if(char(cvWaitKey(15))==27)break;
        }
#endif
    }

#ifdef DEBUG
    thres_window.release();
    cvDestroyWindow("sharpen");
    cvDestroyWindow("src_gauss");
    cvDestroyWindow("open_src");
    cvDestroyWindow("thres_window");
    cvDestroyWindow("erode_out");
    cvDestroyWindow("sepertate_im");
    cvDestroyWindow("single_char");
    cvDestroyWindow("window_add");
#endif
}

void TextDetector::segmentSobMor(cv::Mat &spineGray, vector<Mat> &single_char_vec, int im_num, bool save)
{
    Mat spineGrayTemp = spineGray - 0.5;
    cv::Mat spineAhe;
    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);

    cv::Mat spineShrpen;
    sharpenImage(spineAhe, spineShrpen);
    while(1)
    {
      imshow("sharpen", spineShrpen);
      if(char(cvWaitKey(15))==27)break;
    }

    cv::Mat src_sobel;
    int m_GaussianBlurSize = 5;
    sobelOper(spineShrpen, src_sobel, m_GaussianBlurSize);
    while(1)
    {
      imshow("src_sobel", src_sobel);
      if(char(cvWaitKey(15))==27)break;
    }

//    double minVal,maxVal;
//    Point minLoc,maxLoc;
//    minMaxLoc(src_sobel,&minVal,&maxVal,&minLoc,&maxLoc);
//    double threshold_value = maxVal*0.7;
    double threshold_value = 150;
//    std::cout<<" threshold_value "<<maxVal<<std::endl;
    cv::Mat window_tmp;
//    threshold(src_sobel, window_tmp, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
    threshold(src_sobel, window_tmp, threshold_value, 255, CV_THRESH_BINARY);
//    adaptiveThreshold(spineShrpen, window_tmp,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY, 5, 0);
//    std::cout<<window_tmp<<std::endl;
    while(1)
    {
      imshow("window_tmp", window_tmp);
      if(char(cvWaitKey(15))==27)break;
    }
    //进行open操作
    Mat element_sob = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat open_sob;
    morphologyEx(window_tmp,open_sob,MORPH_OPEN,element_sob);
    while(1)
    {
      imshow("open_sob", open_sob);
      if(char(cvWaitKey(15))==27)break;
    }
    cv::Mat thres_window = open_sob.clone();

    while(1)
    {
      imshow("thres_window", thres_window);
      if(char(cvWaitKey(15))==27)break;
    }

    //进行close操作
    Mat elementDilate = getStructuringElement(MORPH_RECT, Size(5, 15));
    Mat elementErode = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dilate_out,erode_out;
    morphologyEx(thres_window,dilate_out,MORPH_DILATE,elementDilate);
    morphologyEx(dilate_out,erode_out,MORPH_ERODE,elementErode);
    while(1)
    {
      imshow("erode_out", erode_out);
      if(char(cvWaitKey(15))==27)break;
    }

    Mat img_contours;
    erode_out.copyTo(img_contours);
    vector<vector<Point> > contours;
    findContours(img_contours,
                 contours,               // a vector of contours
                 CV_RETR_EXTERNAL,       // retrieve the external contours
                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
    Mat sepertate_im(thres_window.size(),thres_window.depth(),Scalar(255));
    drawContours(sepertate_im,contours,-1,Scalar(0),2);
    while(1)
    {
      imshow("sepertate_im",sepertate_im);
      if(char(cvWaitKey(15))==27)break;
    }

    vector<vector<Point> >::iterator itc = contours.begin();
    vector<Rect> vecRect;



    while (itc != contours.end())
    {
      Rect mr = boundingRect(Mat(*itc));
      Mat auxRoi(thres_window, mr);
      if (/*verifyCharSizes(auxRoi)*/1) vecRect.push_back(mr);
      ++itc;
    }

    ////save single char image after segment
    for(int char_num=0;char_num<vecRect.size();char_num++)
    {
         Mat single_char=thres_window(vecRect.at(char_num));
         single_char_vec.push_back(single_char);
        if(save)
        {
            const char* single_char_folder_ = "../../../src/easyocr/char_img";
            std::stringstream ss(std::stringstream::in | std::stringstream::out);
            ss << single_char_folder_ << "/" << im_num << "_sob" << char_num << ".jpg";
            imwrite(ss.str(),single_char);
        }
        while(1)
        {
          imshow( "single_char", single_char );
          if(char(cvWaitKey(15))==27)break;
        }
    }


    thres_window.release();
    cvDestroyWindow("sharpen");
    cvDestroyWindow("src_sobel");
    cvDestroyWindow("open_sob");
    cvDestroyWindow("thres_window");
    cvDestroyWindow("erode_out");
    cvDestroyWindow("sepertate_im");
    cvDestroyWindow("single_char");
}


void TextDetector::imgQuantize(cv::Mat &src, cv::Mat &dst, double level){
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8U);
    for (int i = 0; i < src.rows; i ++) {
        uchar *data = src.ptr<uchar>(i);
        uchar *data2 = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j ++) {
            if(data[j] <= level)
                data2[j] = 1;
            else
                data2[j] = 2;
                
        }
    }
    
}


