//
//  textDetetcor.hpp
//  textExtracorDemo
//
//  Created by lingLong on 16/8/29.
//  Copyright © 2016年 ling. All rights reserved.
//

#ifndef textDetetcor_hpp
#define textDetetcor_hpp


#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tesseract/basedir.h>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <tesseract/unichar.h>
#include <tesseract/unicharset.h>
#include <leptonica/allheaders.h>
#include <fstream>
#include <bitset>
#include <time.h>
#include "easyocr/config.h"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


using namespace std;
using namespace cv;

#define BIGCHAR 1
#define MEDCHAR 2
#define SMALLCHAR 3

#define PIECEWIDTH 200
#define PIECEHEIGHT 100

#define WHITE 1
#define BLACK 2

#define V_PROJECT 1  //垂直投影（vertical）
#define H_PROJECT 2  //水平投影（horizational）

//#define BIGIMG 1


typedef struct
{
    int begin;
    int end;

}char_range_t;




class TextDetector{
public:
    TextDetector();
//    TextDetector(TextDetecorParams &params, std::string imgDir = "");
    void segmentSobMor(cv::Mat &spineImage, vector<Mat> &single_char_vec, vector<Rect> &vecRect, int im_num, bool save);
    void segmentSrcMor(cv::Mat &spineImage, vector<Mat> &single_char_vec, vector<Rect> vecContoRect, int im_num, bool save);
    void segmentSrcPre(cv::Mat &spineImage);
    void segmentSrcSlide(cv::Mat &spineImage, vector<Mat> &single_char_vec, int char_width, int char_height, int im_num, bool save, int &char_mat_height, int &char_mat_width);
    void segmentSrcProject(cv::Mat &spineGray, vector<Mat> &single_char_vec,int im_num, bool save);
protected:
    //pre-processing
    cv::Mat preProcess(cv::Mat &image);
    void adaptiveHistEqual(cv::Mat &src,cv::Mat &dst,double clipLimit);
    void findKEdgeFirst(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols);
    void findKEdgeLast(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols);
    void sharpenImage(const cv::Mat &image, cv::Mat &result);
    void imgQuantize(cv::Mat &src, cv::Mat &dst, double level);
    bool verifyCharSizes(Mat r);
    int sobelOper(const Mat &in, Mat &out, int blurSize);
    Mat preprocessChar(Mat in);
    void setMorParameters(int char_size);
    void setThreParameters(int char_color);
    int slidingWnd(Mat& src, vector<Mat>& wnd, Size wndSize, double x_percent, double y_percent, int &char_mat_height, int &char_mat_width);
    float findShortestDistance(vector<Point> &contoursA_, vector<Point> &contoursB_, Point &p_a, Point &p_b);
    void removeIsoContour(vector<vector<Point> > &contours, vector<vector<Point> > &contours_remove);
    Rect rectCenterScale(Rect rect, Size size);
    int GetTextProjection(Mat &src, vector<int>& pos, int mode);
    void draw_projection(vector<int>& pos, int mode);
    int GetPeekRange(vector<int> &vertical_pos, vector<char_range_t> &peek_range, int min_thresh, int min_range);
    int draw_main_row(vector<int>& pos,char_range_t &main_peek_rang_v);
private:
//    string imageDirectory;
    double NaN = nan("not a number");
    static const int DEFAULT_GAUSSIANBLUR_SIZE = 5;
    static const int SOBEL_SCALE = 1;
    static const int SOBEL_DELTA = 0;
    static const int SOBEL_DDEPTH = CV_16S;
    static const int SOBEL_X_WEIGHT = 1;
    static const int SOBEL_Y_WEIGHT = 1;
    static const int DEFAULT_MORPH_SIZE_WIDTH = 17;  // 17
    static const int DEFAULT_MORPH_SIZE_HEIGHT = 3;  // 3
    static const int CHAR_SIZE = 20;
    Size src_open_val,src_dilate_val,src_erode_val;
    bool inv_bin = false;
    float connect_dis = 8;
};  /*  class TextDetector */

#endif /* textDetetcor_hpp */
