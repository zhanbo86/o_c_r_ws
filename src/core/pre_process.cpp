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
           connect_dis = 8;
        break;
        case MEDCHAR:
            src_open_val = Size(5, 5);
            src_dilate_val = Size(5, 13);
            src_erode_val = Size(5, 5);
            connect_dis = 3;
        break;
        case SMALLCHAR:
            src_open_val = Size(3, 5);
            src_dilate_val = Size(1, 8);
            src_erode_val = Size(1, 5);
            connect_dis = 2;
        break;
        default:
           std::cout<<"char size input is wrong!!! use default big char parameters."<<std::endl;
           src_open_val = Size(5, 5);
           src_dilate_val = Size(5, 20);
           src_erode_val = Size(5, 5);
           connect_dis = 8;
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

int TextDetector::slidingWnd(Mat& src, vector<Mat>& wnd,Size wndSize, double x_percent, double y_percent,
                             int &char_mat_height,int &char_mat_width)
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
    int rows_count=0;
    int cols_count=0;

    //利用窗口对图像进行遍历
    for (int i = 0; i < src.rows- wndSize.height; i+=y_step)
    {
        rows_count++;
        cols_count = 0;
        for (int j = 0; j < src.cols- wndSize.width; j+=x_step)
        {
            Rect roi(Point(j, i), wndSize);
            Mat ROI = src(roi);
            wnd.push_back(ROI);
            count++;
            cols_count++;

//            cv::Mat idx;
//            findNonZero(ROI, idx);
//            int one_count = (int)idx.total();

//            int zero_count = (int)ROI.total() - one_count;
//            float one_percent = (float)one_count/(float)(one_count+zero_count);
////            std::cout<<"one_count = "<<one_count<<std::endl;
////            std::cout<<"zero_count = "<<zero_count<<std::endl;
////            std::cout<<"one_percent = "<<one_percent<<std::endl;
//            if(one_percent>0.1)
//            {
//                 wnd.push_back(ROI);
//                 count++;
//            }
//            else
//            {
//                single_char_precise.at<uchar>(i,j) = 0;
//            }
        }
    }
    char_mat_height = rows_count;
    char_mat_width = cols_count;

    int64 count2 = getTickCount();
    double time = (count2 - count1) / freq;
    cout << "slide Time=" << time * 100 << "ms"<<endl;
    return count;
}

Rect TextDetector::rectCenterScale(Rect rect, Size size)
{
    rect = rect + size;
    Point pt;
    pt.x = cvRound(size.width/2.0);
    pt.y = cvRound(size.height/2.0);
    return (rect-pt);
}


void TextDetector::removeIsoContour(vector<vector<Point> > &contours)
{
    ////detect contous neiboughbour
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<vector<Point> >::iterator itc2 = contours.begin();
    vector<vector<Point> >::iterator itc_next = contours.begin();
    while (itc != contours.end())
    {
        Rect mr = boundingRect(Mat(*itc));
        Rect mr_3zoom = rectCenterScale(mr,Size(4*mr.width,4*mr.height));

        Mat sepertate_1(Size(500,400),CV_8UC1,Scalar(0));
        rectangle(sepertate_1, mr, Scalar(255, 0, 0), 3);
        Mat sepertate_2(Size(500,400),CV_8UC1,Scalar(0));
        rectangle(sepertate_2, mr_3zoom, Scalar(255, 0, 0), 3);
        itc_next = contours.begin();
        long int mr_cross_acc_width = 0;
        long int mr_cross_acc_height = 0;
        while(itc_next != contours.end())
        {
            if(itc_next==itc)
            {
                itc_next++;
                continue;
            }
            Rect mr_next = boundingRect(Mat(*itc_next));
            Rect mr_cross = mr_3zoom&mr_next;
            mr_cross_acc_width += mr_cross.width;
            mr_cross_acc_height += mr_cross.height;
            itc_next++;
            rectangle(sepertate_2, mr_next, Scalar(255, 0, 0), 3);
            if((mr_cross_acc_height!=0)||(mr_cross_acc_width!=0))
            {
                break;
            }
        }
        if((mr_cross_acc_width==0)&&(mr_cross_acc_height==0))
        {
            itc2 = contours.erase(itc);
            itc = itc2;
            std::cout<<"erase this contour!!!"<<std::endl;
        }
        else
        {
            ++itc;
        }
    }
}

float TextDetector::findShortestDistance(vector<Point> &contoursA_, vector<Point> &contoursB_, Point &p_a, Point &p_b)
{
    float distance=0;
    float min_distance=200;
    vector<Point> contoursA = contoursA_;
    vector<Point> contoursB = contoursB_;
    vector<Point>::iterator itc_a = contoursA.begin();
    vector<Point>::iterator itc_b = contoursB.begin();
    while (itc_a != contoursA.end())
    {
        itc_b = contoursB.begin();
        while (itc_b != contoursB.end())
        {
            distance = sqrt(pow(((*itc_a).x - (*itc_b).x),2)+pow(((*itc_a).y - (*itc_b).y),2));
            if(distance < min_distance)
            {
                min_distance = distance;
                p_a = *itc_a;
                p_b = *itc_b;
            }
            itc_b++;
        }
        itc_a++;
    }
    return min_distance;
}


//segment the spine text
void TextDetector::segmentSrcSlide(cv::Mat &spineGray, vector<Mat> &single_char_vec,
                                   int char_width, int char_height, int im_num, bool save,
                                   int &char_mat_height,int &char_mat_width)
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



//    //// Pass it to Tesseract API
//    tesseract::TessBaseAPI tess;
//    tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
////    tess.SetVariable("tessedit_char_whitelist", "0123456789");
////    tess.SetVariable("classify_bln_numeric_mode", "1");
////    tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);
//    tess.SetImage((uchar*)thresh_src.data, thresh_src.cols, thresh_src.rows, thresh_src.channels(), thresh_src.cols);
//    Boxa* boxes = tess.GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL);
//    printf("Found %d textline image components.\n", boxes->n);
//    for (int i = 0; i < boxes->n; i++){
//        BOX* box = boxaGetBox(boxes, i, L_CLONE);
//        tess.SetRectangle(box->x, box->y, box->w, box->h);
//        char* ocrResult = tess.GetUTF8Text();
//        int conf = tess.MeanTextConf();
//        fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s",
//            i, box->x, box->y, box->w, box->h, conf, ocrResult);
//    }

//    //// Get the text
//    char* out = tess.GetUTF8Text();
//    std::cout << out << std::endl;



    ////slide window in src
    vector<Mat> charWnd;
    int count=slidingWnd(thresh_src, charWnd,Size(char_width, char_height),0.1,0.1,char_mat_height,char_mat_width);
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
//    cv::Mat thres_window = open_src.clone();
    cv::Mat thres_window = thresh_src.clone();


//    //////morphological close
//    Mat elementDilate = getStructuringElement(MORPH_RECT, src_dilate_val);
//    Mat elementErode = getStructuringElement(MORPH_RECT, src_erode_val);
//    Mat dilate_out,erode_out;
//    morphologyEx(thres_window,dilate_out,MORPH_DILATE,elementDilate);
//    morphologyEx(dilate_out,erode_out,MORPH_ERODE,elementErode);
////    erode_out = dilate_out.clone();
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("erode_out", erode_out);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

    ////find contours
    Mat img_contours;
    thres_window.copyTo(img_contours);
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


    ////detect contous neiboughbour
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<vector<Point> >::iterator itc_next = contours.begin();
    int contours_num_a = 0;
    int contours_num_b = 0;
    while (itc != contours.end())
    {
        contours_num_a++;
        itc_next = itc;
        itc_next++;
        Point p_a;
        Point p_b;
        float min_distance = 200;
        while(itc_next != contours.end())
        {
            vector<Point> contoursA = *itc;
            vector<Point> contoursB = *itc_next;
            min_distance = findShortestDistance(contoursA,contoursB,p_a,p_b);
//            std::cout<<"min_distance = "<<min_distance<<std::endl;
//            std::cout<<"p_a = ("<<p_a.x<<","<<p_a.y<<")"<<std::endl;
//            std::cout<<"p_b = ("<<p_b.x<<","<<p_b.y<<")"<<std::endl;
            if(min_distance < 10)
            {
                line(thres_window, p_a, p_b, Scalar(255, 0, 0), 10);
            }
            itc_next++;
            contours_num_b++;
        }
      ++itc;
    }
#ifdef DEBUG
    while(1)
    {
      imshow("thres_window", thres_window);
      if(char(cvWaitKey(15))==27)break;
    }
#endif



//    ////remove noise
//    vector<vector<Point> >::iterator itc = contours.begin();
//    vector<Rect> vecRect;
//    cv::Mat rectangle_show = thres_window.clone();
//    while (itc != contours.end())
//    {
//      Rect mr = boundingRect(Mat(*itc));
//      Mat auxRoi(thres_window, mr);
//      if (/*verifyCharSizes(auxRoi)*/1) vecRect.push_back(mr);
//      ++itc;
//      rectangle(rectangle_show, mr, Scalar(255, 0, 0), 3);
//    }
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("rectangle_show",rectangle_show);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

//    ////save single char image after segment
//    for(int char_num=0;char_num<vecRect.size();char_num++)
//    {
//         Mat single_char_=thres_window(vecRect.at(char_num));
//         Mat single_char;
//         single_char = preprocessChar(single_char_);
//         single_char_vec.push_back(single_char);
//        if(save)
//        {
//            const char* single_char_folder_ = "../../../src/easyocr/char_img";
//            std::stringstream ss(std::stringstream::in | std::stringstream::out);
//            ss << single_char_folder_ << "/" << im_num << "_src" << char_num<<"_"<<rand()<< ".jpg";
//            imwrite(ss.str(),single_char);
//        }
//#ifdef DEBUG
//        while(1)
//        {
//          imshow( "single_char", single_char_ );
//          if(char(cvWaitKey(15))==27)break;
//        }
//#endif
//    }

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
    cvDestroyWindow("rectangle_show");
#endif
}

void TextDetector::segmentSobMor(cv::Mat &spineGray, vector<Mat> &single_char_vec, int im_num, bool save)
{
    int char_size;
#ifdef DEBUG
    printf("please input char size: big is 1, mediate is 2, small is 3\n");
    scanf("%d",&char_size);
#endif
    setMorParameters(char_size);

    ////histequal and shrpen
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


    ////soble
    cv::Mat src_sobel;
    int m_GaussianBlurSize = 5;
    sobelOper(spineShrpen, src_sobel, m_GaussianBlurSize);
    while(1)
    {
      imshow("src_sobel", src_sobel);
      if(char(cvWaitKey(15))==27)break;
    }

    ////threshold
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
      imshow("window_thresh", window_tmp);
      if(char(cvWaitKey(15))==27)break;
    }
    ////进行open操作
    Mat element_sob = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat open_sob;
    morphologyEx(window_tmp,open_sob,MORPH_OPEN,element_sob);
    while(1)
    {
      imshow("open_sob", open_sob);
      if(char(cvWaitKey(15))==27)break;
    }
    cv::Mat thres_window = open_sob.clone();

//    //进行close操作
//    Mat elementDilate = getStructuringElement(MORPH_RECT, Size(5, 15));
//    Mat elementErode = getStructuringElement(MORPH_RECT, Size(5, 5));
//    Mat dilate_out,erode_out;
//    morphologyEx(thres_window,dilate_out,MORPH_DILATE,elementDilate);
//    morphologyEx(dilate_out,erode_out,MORPH_ERODE,elementErode);
//    while(1)
//    {
//      imshow("erode_out", erode_out);
//      if(char(cvWaitKey(15))==27)break;
//    }


    ////find contours
    Mat img_contours;
    thres_window.copyTo(img_contours);
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


    Mat sepertate_im_remove(thres_window.size(),thres_window.depth(),Scalar(255));
    removeIsoContour(contours);
    std::cout<<"it is ok!!!!"<<std::endl;
    drawContours(sepertate_im_remove,contours,-1,Scalar(0),2);
    while(1)
    {
      imshow("sepertate_im_remove",sepertate_im_remove);
      if(char(cvWaitKey(15))==27)break;
    }


    ////detect contous neiboughbour
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<vector<Point> >::iterator itc_next = contours.begin();
    int contours_num_a = 0;
    int contours_num_b = 0;
    while (itc != contours.end())
    {
        contours_num_a++;
        itc_next = itc;
        itc_next++;
        Point p_a;
        Point p_b;
        float min_distance = 200;
        float threshold_distance;
        while(itc_next != contours.end())
        {
            vector<Point> contoursA = *itc;
            vector<Point> contoursB = *itc_next;
            min_distance = findShortestDistance(contoursA,contoursB,p_a,p_b);
//            std::cout<<"min_distance = "<<min_distance<<std::endl;
//            std::cout<<"p_a = ("<<p_a.x<<","<<p_a.y<<")"<<std::endl;
//            std::cout<<"p_b = ("<<p_b.x<<","<<p_b.y<<")"<<std::endl;
            if(min_distance!=0)
            {
                threshold_distance = connect_dis*(1+1.2*pow(abs((float)(p_b.y-p_a.y))/min_distance,2));
            }
            if(min_distance < threshold_distance)
            {
                line(thres_window, p_a, p_b, Scalar(255, 0, 0), 3);
            }
            itc_next++;
            contours_num_b++;
        }
      ++itc;
    }
#ifdef DEBUG
    while(1)
    {
      imshow("thres_window", thres_window);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////find contours again
    Mat img_contours_again;
    thres_window.copyTo(img_contours_again);
    vector<vector<Point> > contours_again;
    findContours(img_contours_again,
                 contours_again,               // a vector of contours
                 CV_RETR_EXTERNAL,       // retrieve the external contours
                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
    Mat sepertate_im_again(thres_window.size(),thres_window.depth(),Scalar(255));
    drawContours(sepertate_im_again,contours_again,-1,Scalar(0),2);
    while(1)
    {
      imshow("sepertate_im_again",sepertate_im_again);
      if(char(cvWaitKey(15))==27)break;
    }

    vector<vector<Point> >::iterator itc_again = contours_again.begin();
    vector<Rect> vecRect;
    while (itc_again != contours_again.end())
    {
      Rect mr = boundingRect(Mat(*itc_again));
      Mat auxRoi(thres_window, mr);
      if (/*verifyCharSizes(auxRoi)*/1) vecRect.push_back(mr);
      ++itc_again;
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
    cvDestroyWindow("sepertate_im_again");
    cvDestroyWindow("single_char");
    cvDestroyWindow("window_thresh");

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


