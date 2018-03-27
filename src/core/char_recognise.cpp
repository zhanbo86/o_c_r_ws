#include "easyocr/core/char_recognise.hpp"

using namespace cv;
using namespace std;
using namespace easyocr;
#include <stdio.h>



CharRecog::CharRecog(const char *chars_folder)
: chars_folder_(chars_folder)
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

int CharRecog::cdata(std::vector<cv::Mat> &matVec)
{
    assert(chars_folder_);

    auto chars_files = Utils::getFiles(chars_folder_);
    if (chars_files.size() == 0) {
      fprintf(stdout, "No file found in the train folder!\n");
      fprintf(stdout, "You should create a folder named \"tmp\" in easyocr main folder.\n");
      fprintf(stdout, "Copy train data folder(like \"ann\") under \"tmp\". \n");
      return -1;
    }
    size_t char_size = chars_files.size();
    fprintf(stdout, ">> img count: %d \n", int(char_size));

    for (auto file : chars_files) {
      std::cout<<"image name = "<<file<<std::endl;
      auto img = cv::imread(file, 0);  // a grayscale image
      matVec.push_back(img);
    }
    fprintf(stdout, ">> img collect count: %d \n", (int)matVec.size());
    return 0;
}

int CharRecog::charRecognise()
{
    //load raw img data.
    std::cout<<"load img from raw_img file."<<std::endl;
    int number_for_count = 350;
    std::vector<cv::Mat> matVec;
    matVec.reserve(number_for_count);
    if(cdata(matVec)!=0)
    {
        return -1;
    };

    //preproceed imgs
    std::cout << "preproceed imgs." << std::endl;
    for(int im_num=0;im_num<matVec.size();im_num++)
    {
        printf("image num %d\n",im_num);
        ////reset for every image
        num_p_1 = 0;
        x_int_1 = 0;
        y_int_1 = 0;
        capture_1 = false;

        //// Load image and resize to 1280*1024,source img is 2448*2048
        Mat im = matVec.at(im_num);
#ifdef BIGIMG
        Size low_res = cv::Size((int)(im.size().width/2),(int)(im.size().height/2));
#else
        Size low_res = cv::Size(PIECEWIDTH,PIECEHEIGHT);
//        Size low_res = cv::Size(im.cols,im.rows);
#endif
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
#ifdef BIGIMG
    cvSetMouseCallback("source",my_mouse_callback_1,NULL);
#endif


    ////pick ocr piece
    cv::Mat ocr_piece;
    #ifdef BIGIMG
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
    //    num_p_1 = 0;
    //    x_int_1 = 0;
    //    y_int_1 = 0;
    //    capture_1 = false;
    //    imshow( "pick_ocr", ocr_piece );
    //    cvSetMouseCallback("pick_ocr",my_mouse_callback_1,NULL);
    //    while(num_p_1!=4)
    //    {
    //      imshow( "pick_ocr", ocr_piece );
    //      if(char(cvWaitKey(15))==27)break;
    //    }

    //    int char_width = 0;
    //    int char_height = 0;
    //    if(num_p_1==4)
    //    {
    //      char_width = x_temp_1[1] - x_temp_1[0];
    //      char_height = y_temp_1[2] - y_temp_1[1];
    //    }
        cvDestroyWindow("source");
        cvDestroyWindow("pick_ocr");
    #else
        ocr_piece = img_100.clone();
    #endif
//        while(1)
//        {
//          imshow( "pick_ocr", ocr_piece );
//          if(char(cvWaitKey(15))==27)break;
//        }
//        cvDestroyWindow("pick_ocr");






        ////pre-process image
        clock_t a=clock();
        TextDetector detector;
        vector<Mat> single_char_vec;
        single_char_vec.clear();
        int char_mat_height,char_mat_width;
        vector<Rect> vecContoRect;
//        detector.segmentSrcSlide(ocr_piece, single_char_vec,char_width,char_height,0,true,char_mat_height,char_mat_width);
//        Mat single_char_precise(char_mat_height,char_mat_width, CV_8UC1);
//        detector.segmentSrcMor(ocr_piece, single_char_vec,im_num,true);
//        detector.segmentSrcPre(ocr_piece);
//        detector.segmentSobMor(ocr_piece, single_char_vec,vecContoRect,im_num,false);
//        detector.segmentSrcMor(ocr_piece, single_char_vec,vecContoRect,im_num,false);
         detector.segmentSrcProject(ocr_piece, single_char_vec);


         //// identifing single characters
         std::string license;
     #ifdef DEBUG
         std::cout<<"single_char_vec.size = "<<single_char_vec.size()<<std::endl;
     #endif
         int char_num_rem = 0;
         for(;char_num_rem<single_char_vec.size();char_num_rem++)
         {
            Mat single_char;
            single_char = single_char_vec.at(char_num_rem);



             const char* single_char_folder_ = "../../../src/easyocr/rec_char_img";
             std::stringstream ss(std::stringstream::in | std::stringstream::out);
//             ss << single_char_folder_ << "/" << im_num << "_src" << char_num<<"_"<<rand()<< ".jpg";
             ss << single_char_folder_ << "/" << char_num_rem<<".jpg";
             imwrite(ss.str(),single_char);


     #ifdef DEBUG
             while(1)
             {
               imshow("single_char",single_char);
               if(char(cvWaitKey(15))==27)break;
             }
             cvDestroyWindow("single_char");
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
             while(1)
             {
               imshow( "single_char", single_char );
               if(char(cvWaitKey(15))==27)break;
             }
             cvDestroyWindow("single_char");
     #endif
         }
         std::cout << "PlantIdentify: " << license << std::endl;
         clock_t b=clock();
         cout<<"time cost = "<<1000*(double)(b - a) / CLOCKS_PER_SEC<<" ms "<<endl;
         while(1)
         {
           imshow( "pick_ocr", ocr_piece );
           if(char(cvWaitKey(15))==27)break;
         }

         std::cout<<"char_num = "<<char_num_rem<<std::endl;
//         for(int i=0;i<=char_num_rem;i++)
//         {
//             const char* single_char_folder_ = "../../../src/easyocr/rec_char_img";
//             std::stringstream ss(std::stringstream::in | std::stringstream::out);
//             ss << single_char_folder_ << "/" << i<<".jpg";
//             char* file_char;
//             ss>>file_char;
//             std::cout<<"i = "<<i<<std::endl;
//             remove(file_char);

////             DeleteFile((LPCTSTR)ss.c_str());
//         }
         cvDestroyWindow("pick_ocr");
         std::cout<<"***********************************************************"<<std::endl;

    }
    return 0;
}


