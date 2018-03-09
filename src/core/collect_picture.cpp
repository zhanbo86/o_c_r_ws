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
    int number_for_count = 2050;
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
        std::cout<<"images num = "<<im_num<<std::endl;
        ////reset for every image
        num_p_2 = 0;
        x_int_2 = 0;
        y_int_2 = 0;
        capture_2 = false;

        //// Load image and resize to 1280*1024,source img is 2448*2048
        Mat im = matVec.at(im_num);
#ifdef BIGIMG
        Size low_res = cv::Size((int)(im.size().width/2),(int)(im.size().height/2));
#else
        Size low_res = cv::Size(PIECEWIDTH,PIECEHEIGHT);
#endif
        Mat img_100(low_res,im.depth(),1);
        if (im.empty())
        {
            std::cout << "Cannot open source image!" << std::endl;
            std::cout<<"total images num = "<<im_num<<std::endl;
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
//        while(1)
//        {
//          imshow( "pick_ocr", ocr_piece );
//          if(char(cvWaitKey(15))==27)break;
//        }
        cvDestroyWindow("pick_ocr");
    #endif


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





         //// save single characters
         std::string license;
     #ifdef DEBUG
         std::cout<<"single_char_vec.size = "<<single_char_vec.size()<<std::endl;
     #endif
         for(int char_num=0;char_num<single_char_vec.size();char_num++)
         {
            Mat single_char;
            single_char = single_char_vec.at(char_num);
             const char* single_char_folder_ = "../../../src/easyocr/char_img";
             std::stringstream ss(std::stringstream::in | std::stringstream::out);
             ss << single_char_folder_ << "/" << im_num << "_src" << char_num<<"_"<<rand()<< ".jpg";
             imwrite(ss.str(),single_char);
         }
    }

    return 0;
}


