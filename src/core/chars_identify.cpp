#include "easyocr/core/chars_identify.h"
#include "easyocr/core/character.hpp"
#include "easyocr/core/core_func.h"
#include "easyocr/core/feature.h"
#include "easyocr/core/params.h"
#include "easyocr/config.h"

using namespace cv;

namespace easyocr {

CharsIdentify* CharsIdentify::instance_ = nullptr;

CharsIdentify* CharsIdentify::instance() {
  if (!instance_) {
    instance_ = new CharsIdentify;
  }
  return instance_;
}

CharsIdentify::CharsIdentify() {
  LOAD_ANN_MODEL(ann_, kDefaultAnnPath);

}

void CharsIdentify::LoadModel(std::string path) {
  if (path != std::string(kDefaultAnnPath)) {
    if (!ann_->empty())
      ann_->clear();
    LOAD_ANN_MODEL(ann_, path);
  }
}



void CharsIdentify::LoadChineseMapping(std::string path) {
  kv_->clear();
  kv_->load(path);
}

void CharsIdentify::classify(cv::Mat featureRows, std::vector<int>& out_maxIndexs,
                             std::vector<float>& out_maxVals)
{
  int rowNum = featureRows.rows;
  cv::Mat output(rowNum, kCharactersNumber, CV_32FC1);
  ann_->predict(featureRows, output);
  for (int output_index = 0; output_index < rowNum; output_index++)
  {
    Mat output_row = output.row(output_index);
    int result = 0;
    float maxVal = -2.f;
    result = 0;
    for (int j = 0; j < kCharactersNumber; j++)
    {
        float val = output_row.at<float>(j);
        // std::cout << "j:" << j << "val:" << val << std::endl;
        if (val > maxVal)
        {
          maxVal = val;
          result = j;
        }
    }
    out_maxIndexs[output_index] = result;
    out_maxVals[output_index] = maxVal;
  }
}


void CharsIdentify::classify(std::vector<CCharacter>& charVec){
  size_t charVecSize = charVec.size();

  if (charVecSize == 0)
    return;

  Mat featureRows;
  for (size_t index = 0; index < charVecSize; index++) {
    Mat charInput = charVec[index].getCharacterMat();
    Mat feature = charFeatures(charInput, kPredictSize);
    featureRows.push_back(feature);
  }

  cv::Mat output(charVecSize, kCharactersNumber, CV_32FC1);
  ann_->predict(featureRows, output);

  for (size_t output_index = 0; output_index < charVecSize; output_index++)
  {
    CCharacter& character = charVec[output_index];
    Mat output_row = output.row(output_index);

    int result = 0;
    float maxVal = -2.f;
    std::string label = "";
    result = 0;
    for (int j = 0; j < kCharactersNumber; j++)
    {
       float val = output_row.at<float>(j);
       //std::cout << "j:" << j << "val:" << val << std::endl;
       if (val > maxVal)
       {
          maxVal = val;
          result = j;
       }
     }
   label = std::make_pair(kChars[result], kChars[result]).second;
    /*std::cout << "result:" << result << std::endl;
    std::cout << "maxVal:" << maxVal << std::endl;*/
    character.setCharacterScore(maxVal);
    character.setCharacterStr(label);
  }
}


//void CharsIdentify::classifyChineseGray(std::vector<CCharacter>& charVec){
//  size_t charVecSize = charVec.size();
//  if (charVecSize == 0)
//    return;

//  Mat featureRows;
//  for (size_t index = 0; index < charVecSize; index++) {
//    Mat charInput = charVec[index].getCharacterMat();
//    cv::Mat feature;
//    extractFeature(charInput, feature);
//    featureRows.push_back(feature);
//  }

//  cv::Mat output(charVecSize, kChineseNumber, CV_32FC1);
//  annGray_->predict(featureRows, output);

//  for (size_t output_index = 0; output_index < charVecSize; output_index++) {
//    CCharacter& character = charVec[output_index];
//    Mat output_row = output.row(output_index);
//    bool isChinese = true;

//    float maxVal = -2;
//    int result = 0;

//    for (int j = 0; j < kChineseNumber; j++) {
//      float val = output_row.at<float>(j);
//      //std::cout << "j:" << j << "val:" << val << std::endl;
//      if (val > maxVal) {
//        maxVal = val;
//        result = j;
//      }
//    }

//    // no match
//    if (-1 == result) {
//      result = 0;
//      maxVal = 0;
//      isChinese = false;
//    }

//    auto index = result + kCharsTotalNumber - kChineseNumber;
//    const char* key = kChars[index];
//    std::string s = key;
//    std::string province = kv_->get(s);

//    /*std::cout << "result:" << result << std::endl;
//    std::cout << "maxVal:" << maxVal << std::endl;*/

//    character.setCharacterScore(maxVal);
//    character.setCharacterStr(province);
//    character.setIsChinese(isChinese);
//  }
//}


int CharsIdentify::classify(cv::Mat f, float& maxVal, bool isAlphabet){
  int result = 0;

  cv::Mat output(1, kCharactersNumber, CV_32FC1);
  ann_->predict(f, output);

  maxVal = -2.f;
    if (!isAlphabet) {
      result = 0;
      for (int j = 0; j < kCharactersNumber; j++) {
        float val = output.at<float>(j);
//         std::cout << "j:" << j << " ,  val:" << val << std::endl;
        if (val > maxVal) {
          maxVal = val;
          result = j;
        }
      }
    }
//    else {
//      result = 0;
//      // begin with 11th char, which is 'A'
//      for (int j = 10; j < kCharactersNumber; j++) {
//        float val = output.at<float>(j);
//        // std::cout << "j:" << j << "val:" << val << std::endl;
//        if (val > maxVal) {
//          maxVal = val;
//          result = j;
//        }
//      }
//    }
  std::cout << "maxVal:" << maxVal << std::endl;
  return result;
}

bool CharsIdentify::isCharacter(cv::Mat input, std::string& label, float& maxVal, bool isChinese) {
  cv::Mat feature = charFeatures(input, kPredictSize);
  auto index = static_cast<int>(classify(feature, maxVal, isChinese));

  if (isChinese) {
    //std::cout << "maxVal:" << maxVal << std::endl;
  }

  float chineseMaxThresh = 0.2f;

  if (maxVal >= 0.9 || (isChinese && maxVal >= chineseMaxThresh)) {
    if (index < kCharactersNumber) {
      label = std::make_pair(kChars[index], kChars[index]).second;
    }
    else {
      const char* key = kChars[index];
      std::string s = key;
      std::string province = kv_->get(s);
      label = std::make_pair(s, province).second;
    }
    return true;
  }
  else
    return false;
}


std::pair<std::string, std::string> CharsIdentify::identify(cv::Mat input, float &maxValue_,bool isAlphabet) {
  cv::Mat feature = charFeatures(input, kPredictSize);
  float maxVal = -2;
  auto index = static_cast<int>(classify(feature, maxVal,isAlphabet));
  maxValue_ = maxVal;
  if (index < kCharactersNumber) {
      if(maxVal>0.90)
      {
          return std::make_pair(kChars[index], kChars[index]);
      }
      else
      {
          return std::make_pair("NAN", "NAN");
      }
  }
  else {
    const char* key = kChars[index];
    std::string s = key;
    std::string province = kv_->get(s);
    return std::make_pair(s, province);
  }
}

int CharsIdentify::identify(std::vector<cv::Mat> inputs, std::vector<std::pair<std::string, std::string>>& outputs,
                            std::vector<bool> isChineseVec) {
  Mat featureRows;
  size_t input_size = inputs.size();
  for (size_t i = 0; i < input_size; i++) {
    Mat input = inputs[i];
    cv::Mat feature = charFeatures(input, kPredictSize);
    featureRows.push_back(feature);
  }

  std::vector<int> maxIndexs;
  std::vector<float> maxVals;
  classify(featureRows, maxIndexs, maxVals);

  for (size_t row_index = 0; row_index < input_size; row_index++) {
    int index = maxIndexs[row_index];
    if (index < kCharactersNumber) {
      outputs[row_index] = std::make_pair(kChars[index], kChars[index]);
    }
    else {
      const char* key = kChars[index];
      std::string s = key;
      std::string province = kv_->get(s);
      outputs[row_index] = std::make_pair(s, province);
    }
  }
  return 0;
}
}
