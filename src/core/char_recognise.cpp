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
    Mat im = imread("/home/zb/BoZhan/ocr_ws/0104PCB/6.bmp",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
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

    ////choice char width and height
    num_p_1 = 0;
    x_int_1 = 0;
    y_int_1 = 0;
    capture_1 = false;
    imshow( "pick_ocr", ocr_piece );
    cvSetMouseCallback("pick_ocr",my_mouse_callback_1,NULL);
    while(num_p_1!=4)
    {
      imshow( "pick_ocr", ocr_piece );
      if(char(cvWaitKey(15))==27)break;
    }

    int char_width = 0;
    int char_height = 0;
    if(num_p_1==4)
    {
      char_width = x_temp_1[1] - x_temp_1[0];
      char_height = y_temp_1[2] - y_temp_1[1];
    }
    cvDestroyWindow("source");
    cvDestroyWindow("pick_ocr");


    ////pre-process image
    clock_t a=clock();
    TextDetector detector;
    vector<Mat> single_char_vec;
    single_char_vec.clear();
    int char_mat_height = 0;
    int char_mat_width = 0;
//    detector.segmentSrcSlide(ocr_piece, single_char_vec,char_width,char_height,0,true,char_mat_height,char_mat_width);
//    Mat single_char_precise(char_mat_height,char_mat_width, CV_8UC1);
//    detector.segmentSrcMor(ocr_piece, single_char_vec,0,false);
    detector.segmentSobMor(ocr_piece, single_char_vec,0,false);
//    std::cout<<"ocr_piece_size = "<<ocr_piece.rows<<" * "<<ocr_piece.cols<<std::endl;
//    std::cout<<"char_width = "<<char_width<<" , "<<"char_height = "<<char_height<<std::endl;
//    std::cout<<"single_char_amount = "<<single_char_vec.size()<<std::endl;
//     std::cout<<"char_mat_height = "<<char_mat_height<<" , "<<"char_mat_width = "<<char_mat_width<<std::endl;


//    //// slide window identifing single characters
//    std::string license;
//#ifdef DEBUG
//    std::cout<<"single_char_vec.size = "<<single_char_vec.size()<<std::endl;
//#endif
//    for(int i=0;i<single_char_vec.size();i++)
//    {
//       Mat single_char;
//       single_char = single_char_vec.at(i);
//#ifdef DEBUG
//        while(1)
//        {
//          imshow("single_char",single_char);
//          if(char(cvWaitKey(15))==27)break;
//        }
//        std::cout << "chars_identify" << std::endl;
//#endif
//        cv::Mat idx;
//        findNonZero(single_char, idx);
//        int one_count = (int)idx.total();
//        int zero_count = (int)single_char.total() - one_count;
//        float one_percent = (float)one_count/(float)(one_count+zero_count);
////            std::cout<<"one_count = "<<one_count<<std::endl;
////            std::cout<<"zero_count = "<<zero_count<<std::endl;
////            std::cout<<"one_percent = "<<one_percent<<std::endl;
//        int rows_ = i/char_mat_width;
//        int cols_ = i - rows_*char_mat_width;
//        if(one_percent>0.1)
//        {
//            auto block = single_char;
//            float maxValue_;
//            int maxValue_int;
//            auto character = CharsIdentify::instance()->identify(block, maxValue_,false);
//            if(maxValue_>1.0)
//            {
//                maxValue_int = 255;
//            }
//            else if(maxValue_<0.0)
//            {
//                maxValue_int = 0;
//            }
//            else
//            {
//                maxValue_int = (int)(maxValue_*255);
//            }
////            std::cout<<"maxValue_int = "<<maxValue_int<<std::endl;
////            std::cout<<"rows = "<<rows_<<" , "<<"cols = "<<cols_<<std::endl;


//            if(character.second!="NAN")
//            {
//                license.append(character.second);
//                single_char_precise.at<uchar>(rows_,cols_) = maxValue_int;
//            }
//            else
//            {
//                 single_char_precise.at<uchar>(rows_,cols_) = 0;
//            }
//        }
//        else
//        {
//            single_char_precise.at<uchar>(rows_,cols_) = 0;
//        }


//#ifdef DEBUG
//        std::cout << "CharIdentify: " << character.second << std::endl;
//#endif
//    }
//    std::cout<<"ocr_piece_size = "<<ocr_piece.rows<<" * "<<ocr_piece.cols<<std::endl;
//    std::cout<<"char_width = "<<char_width<<" , "<<"char_height = "<<char_height<<std::endl;
//    std::cout<<"single_char_amount = "<<single_char_vec.size()<<std::endl;
//    std::cout<<"char_mat_height = "<<char_mat_height<<" , "<<"char_mat_width = "<<char_mat_width<<std::endl;
//    std::cout << "plateIdentify: " << license << std::endl;

//    Mat single_char_precise_zoom;
//    resize(single_char_precise,single_char_precise_zoom,cv::Size((int)(ocr_piece.size().width),(int)(ocr_piece.size().height)));
//    while(1)
//    {
//      imshow( "single_char_precise", single_char_precise_zoom );
//      imshow( "pick_ocr", ocr_piece );
//      if(char(cvWaitKey(15))==27)break;
//    }
//    cvDestroyWindow("single_char_precise");
//    cvDestroyWindow("pick_ocr");




    //// identifing single characters
    std::string license;
#ifdef DEBUG
    std::cout<<"single_char_vec.size = "<<single_char_vec.size()<<std::endl;
#endif
    for(int i=0;i<single_char_vec.size();i++)
    {
       Mat single_char;
       single_char = single_char_vec.at(i);
#ifdef DEBUG
        while(1)
        {
          imshow("single_char",single_char);
          if(char(cvWaitKey(15))==27)break;
        }
#endif
        auto block = single_char;
        float maxValue_;
        auto character = CharsIdentify::instance()->identify(block, maxValue_,false);
        if(character.second!="NAN")
        {
            license.append(character.second);
        }

#ifdef DEBUG
        std::cout << "CharIdentify: " << character.second << std::endl;
#endif
    }
    std::cout << "PlantIdentify: " << license << std::endl;
    clock_t b=clock();
    cout<<"time cost = "<<1000*(double)(b - a) / CLOCKS_PER_SEC<<" ms "<<endl;

    return 0;
}


