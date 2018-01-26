#ifndef CHAR_RECOGNISE_HPP
#define CHAR_RECOGNISE_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "pre_process.hpp"
#include "easyocr/core/chars_identify.h"



class CharRecog
{
public:
    CharRecog();
    ~CharRecog();
    int charRecognise();
    Mat preprocessChar(Mat in);


    static const int CHAR_SIZE = 20;

private:

};

#endif  // CHAR_RECOGNISE_HPP
