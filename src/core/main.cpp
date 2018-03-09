#include "easyocr.h"
#include "easyocr/util/switch.hpp"
#include "easyocr/core/char_recognise.hpp"
#include "easyocr/core/collect_picture.hpp"
#include "easyocr/util/util.h"
#include "easyocr/util/kv.h"


int main(int argc, const char* argv[])
{
  std::shared_ptr<easyocr::Kv> kv(new easyocr::Kv);
  kv->load("../../../src/easyocr/resources/text/chinese_mapping");
  bool isExit = false;
  while (!isExit) {
    easyocr::Utils::print_file_lines("../../../src/easyocr/resources/text/main_menu");
    std::cout << kv->get("make_a_choice") << ":";

    int select = -1;
    bool isRepeat = true;
    while (isRepeat) {
      std::cin >> select;
      isRepeat = false;
      switch (select) {
        case 1:
          {
             CollectPic picture_ocr("../../../src/easyocr/raw_img");
             picture_ocr.imgCollect();
          }
          break;
        case 2:
          std::cout << "Run \"demo ann\" for more usage." << std::endl;
          {
            easyocr::AnnTrain ann("../../../src/easyocr/train_set/char2", "../../../src/easyocr/train_set/ann.xml");
            ann.train();
          }
          break;
        case 3:
          {
             CharRecog ocr_recog("../../../src/easyocr/bometTest");
             ocr_recog.charRecognise();
          }
          break;
        case 4:
          isExit = true;
          break;
        default:
          std::cout << kv->get("input_error") << ":";
          isRepeat = true;
          break;
      }
    }
  }
  return 0;
}
