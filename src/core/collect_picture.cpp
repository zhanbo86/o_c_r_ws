#include "easyocr/core/collect_picture.hpp"

using namespace cv;
using namespace std;
using namespace easyocr;


CollectPic::CollectPic(const char *chars_folder)
: chars_folder_(chars_folder)
{
}

CollectPic::~CollectPic() {}


int num_p_2 = 0;
int x_int_2 = 0;
int y_int_2 = 0;
std::vector<int> x_temp_2(4);
std::vector<int> y_temp_2(4);
bool capture_2 = false;
void my_mouse_callback_2(int event,int x,int y,int flags,void* param)
{
  switch (event) {
  case CV_EVENT_LBUTTONDOWN:{
    capture_2 = true;
    x_int_2 = x;
    y_int_2 = y;
//    std::cout<<"x_int = "<<x_int_2<<"\t"<<"y_int = "<<y_int_2<<std::endl;
//    std::cout<<"capture_ = "<<capture_2<<std::endl;
  }
  break;
  case CV_EVENT_LBUTTONUP:{
    if(capture_2)
    {
      capture_2 = false;
//      std::cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<std::endl;
      x_temp_2.at(num_p_2) = x_int_2;
      y_temp_2.at(num_p_2) = y_int_2;
      num_p_2++;
//      std::cout<<"num_p="<<num_p<<"\t"<<"capture_ = "<<capture_<<std::endl;
//      std::cout<<"x_[num_p] = "<<x_temp.at(num_p-1)<<"\t"<<"y_[num_p] = "<<y_temp.at(num_p-1)<<std::endl;
    }
  }
  break;
  }
}


int CollectPic::cdata(std::vector<cv::Mat> &matVec)
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
      auto img = cv::imread(file, 0);  // a grayscale image
      matVec.push_back(img);
    }
    fprintf(stdout, ">> img collect count: %d \n", (int)matVec.size());
    return 0;
}



int CollectPic::imgCollect()
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
        ////reset for every image
        num_p_2 = 0;
        x_int_2 = 0;
        y_int_2 = 0;
        capture_2 = false;

        //// Load image and resize to 1280*1024,source img is 2448*2048
        Mat im = matVec.at(im_num);
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
        cvSetMouseCallback("source",my_mouse_callback_2,NULL);


        ////pick ocr piece
        cv::Mat ocr_piece;
        while(num_p_2!=4)
        {
          imshow( "source", img_100 );
          cvWaitKey(10);
          //          if(char(cvWaitKey(15))==27)break;
        }
        if(num_p_2==4)
        {
          std::vector<Point2f> obj_corners(4);
          obj_corners[0] = Point( x_temp_2[0], y_temp_2[0] );
          obj_corners[1] = Point( x_temp_2[1], y_temp_2[1] );
          obj_corners[2] = Point( x_temp_2[2], y_temp_2[2] );
          obj_corners[3] = Point( x_temp_2[3], y_temp_2[3] );
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
//          std::cout<<"ocr_piece size "<<ocr_piece.rows<<" * "<<ocr_piece.cols<<std::endl;
          imshow( "pick_ocr", ocr_piece );
//          imshow( "source", img_100 );
          if(char(cvWaitKey(15))==27)break;
        }
        cvDestroyWindow("source");
        cvDestroyWindow("pick_ocr");


        ////pre-process image
        TextDetector detector;
        vector<Mat> single_char_vec;
        single_char_vec.clear();
        detector.segmentTextSrc(ocr_piece, single_char_vec,im_num,true);
//        detector.segmentTextSob(ocr_piece, single_char_vec,im_num,true);

//        ////segment character
//        CvRect rect;
//        vector<CvRect> rect_vec;
//        rect_vec.clear();
//        vector <vector<Point>>contours;
//        findContours(proceed_ocr,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
//        Mat sepertate_im(proceed_ocr.size(),proceed_ocr.depth(),Scalar(255));
//        drawContours(sepertate_im,contours,-1,Scalar(0),2);
//        while(1)
//        {
//          imshow("sepertate_im",sepertate_im);
//          if(char(cvWaitKey(15))==27)break;
//        }

//        for(int i=0;i<(int)contours.size();i++){
//            double g_dConArea = contourArea(contours.at(i));
////            cout<<"coutour area = "<<g_dConArea<<endl;
//            if(g_dConArea>10)
//            {
//                rect = boundingRect(contours.at(i));
//                rect_vec.push_back(rect);
//            }
//        }

//        ////save single char image after segment
//        for(int char_num=0;char_num<rect_vec.size();char_num++)
//        {
////            std::string file_save_jpg;
////            std::stringstream ss_save;
////            ss_save<<im_num<<"_"<<char_num<<".jpg";
////            ss_save>>file_save_jpg;

//            const char* single_char_folder_ = "../../../src/easyocr/char_img";
//            std::stringstream ss(std::stringstream::in | std::stringstream::out);
//            ss << single_char_folder_ << "/" << im_num << "_" << char_num << ".jpg";
////            imwrite(ss.str(), simg);

//            Mat single_char=proceed_ocr(rect_vec.at(char_num));
//            imwrite(ss.str(),single_char);
//            while(1)
//            {
//              imshow( "single_char", single_char );
//              if(char(cvWaitKey(15))==27)break;
//            }
//        }
    }

    return 0;
}

