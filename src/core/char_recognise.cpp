#include "easyocr/core/char_recognise.hpp"

using namespace cv;
using namespace std;
using namespace easyocr;


CharRecog::CharRecog()
{
}

CharRecog::~CharRecog() {}

int num_p_1 = 0;
int x_int_1 = 0;
int y_int_1 = 0;
std::vector<int> x_temp_1(4);
std::vector<int> y_temp_1(4);
bool capture_1 = false;
void my_mouse_callback_1(int event,int x,int y,int flags,void* param)
{
  switch (event) {
  case CV_EVENT_LBUTTONDOWN:{
    capture_1 = true;
    x_int_1 = x;
    y_int_1 = y;
//    std::cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<std::endl;
//    std::cout<<"capture_ = "<<capture_<<std::endl;
  }
  break;
  case CV_EVENT_LBUTTONUP:{
    if(capture_1)
    {
      capture_1 = false;
//      std::cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<std::endl;
      x_temp_1.at(num_p_1) = x_int_1;
      y_temp_1.at(num_p_1) = y_int_1;
      num_p_1++;
//      std::cout<<"num_p="<<num_p<<"\t"<<"capture_ = "<<capture_<<std::endl;
//      std::cout<<"x_[num_p] = "<<x_temp.at(num_p-1)<<"\t"<<"y_[num_p] = "<<y_temp.at(num_p-1)<<std::endl;

    }
  }
  break;
  }
}



Mat CharRecog::preprocessChar(Mat in) {
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

int CharRecog::charRecognise()
{
    ////reset for every image
    num_p_1 = 0;
    x_int_1 = 0;
    y_int_1 = 0;
    capture_1 = false;

    //// Load image and resize to 1280*1024,source img is 2448*2048
    Mat im = imread("/home/zb/BoZhan/ocr_ws/0104PCB/1.bmp",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
    Size low_res = cv::Size((int)(im.size().width/2),(int)(im.size().height/2));
    Mat img_100(low_res,im.depth(),1);
    if (im.empty())
    {
        std::cout << "Cannot open source image!" << std::endl;
        return -1;
    }
    std::cout<<"im source width = "<<im.size().width<<" , "<<" height = "<<im.size().height
             <<" , "<<"depth = "<<im.depth()<<" , "<<"channel = "<<im.channels()<<std::endl;
    //    cv::Mat gray;
    //    cv::cvtColor(im, gray, CV_BGR2GRAY);
    cv::resize(im,img_100,low_res,0,0,CV_INTER_LINEAR);
    std::cout<<"img width = "<<img_100.size().width<<" , "<<" height = "<<img_100.size().height
             <<" , "<<"depth = "<<img_100.depth()<<" , "<<"channel = "<<img_100.channels()<<std::endl;
    imshow( "source", img_100 );
    cvSetMouseCallback("source",my_mouse_callback_1,NULL);


    ////pick ocr piece
    cv::Mat ocr_piece;
    while(num_p_1!=4)
    {
      imshow( "source", img_100 );
      cvWaitKey(10);
      //          if(char(cvWaitKey(15))==27)break;
    }
    clock_t a=clock();
    if(num_p_1==4)
    {
      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = Point( x_temp_1[0], y_temp_1[0] );
      obj_corners[1] = Point( x_temp_1[1], y_temp_1[1] );
      obj_corners[2] = Point( x_temp_1[2], y_temp_1[2] );
      obj_corners[3] = Point( x_temp_1[3], y_temp_1[3] );
      Rect roi_rect = Rect(obj_corners[0].x,obj_corners[0].y,
                           obj_corners[1].x-obj_corners[0].x,
                           obj_corners[3].y-obj_corners[0].y);
      img_100(roi_rect).copyTo(ocr_piece);
      line( img_100 , obj_corners[0], obj_corners[1], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
      line( img_100 , obj_corners[1], obj_corners[2], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
      line( img_100 , obj_corners[2], obj_corners[3], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
      line( img_100 , obj_corners[3], obj_corners[0], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
    }
    while(1)
    {
      imshow( "pick_ocr", ocr_piece );
      imshow( "source", img_100 );
      if(char(cvWaitKey(15))==27)break;
    }
    cvDestroyWindow("source");


    ////pre-process image
    TextDetector detector;
    vector<Mat> single_char_vec;
    single_char_vec.clear();
    detector.segmentTextSrc(ocr_piece, single_char_vec,0,false);
    detector.segmentTextSob(ocr_piece, single_char_vec,0,false);


    //// identifing single characters.
    std::string license;
    std::cout<<"single_char_vec.size = "<<single_char_vec.size()<<std::endl;
    for(int i=0;i<single_char_vec.size();i++)
    {
       Mat single_char_;
       single_char_ = single_char_vec.at(i);
       Mat single_char;
       single_char = preprocessChar(single_char_);
       std::cout<<"single_char = "<<single_char.rows<<" * "<<single_char.cols<<std::endl;
        while(1)
        {
          imshow("single_char",single_char);
          if(char(cvWaitKey(15))==27)break;
        }
        std::cout << "chars_identify" << std::endl;
        auto block = single_char;
        auto character = CharsIdentify::instance()->identify(block, false);
        license.append(character.second);
        std::cout << "CharIdentify: " << character.second << std::endl;
    }
    std::cout << "plateIdentify: " << license << std::endl;

    clock_t b=clock();
    cout<<"time cost = "<<1000*(double)(b - a) / CLOCKS_PER_SEC<<" ms "<<endl;

    cvDestroyWindow("pick_ocr");
    cvDestroyWindow("single_char");
    return 0;
}

