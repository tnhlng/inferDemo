//
// Created by Admin on 2024/7/8.
//

#ifndef DEMO_UTIL_H
#define DEMO_UTIL_H
#include <opencv2/core/core.hpp>
#include <filesystem>
#include "json/nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

template<typename T>
void imageChannelCopyTo(const cv::Mat& img, std::vector<T>& vecData) {
    // �ָ�����ͨ��
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    vecData.resize(int(3 * img.rows * img.cols));

    int offset = 0;
    for (int c = 0; c < 3; ++c)
    {
        // ����תΪ����
        //cv::Mat temp;
        //channels[c].convertTo(temp, CV_32F);

        // ���Ƶ�vector<float>
        std::memcpy(vecData.data() + offset, channels[c].data, channels[c].total() * sizeof(T));
        offset += channels[c].total();
    }
};

template<typename T>
void imagePixelCopyTo(const cv::Mat& img, std::vector<T>& vecData) {
    size_t pixels = img.total();
    size_t channel = img.channels();
    size_t elemSize = img.elemSize();
    size_t bytes = img.total() * img.elemSize();

    vecData.resize(pixels * channel);
    if (img.isContinuous()) {
        memcpy(vecData.data(), img.ptr<T>(0), bytes);
    }
    else {
        size_t colSize = int(img.cols * img.channels() * img.elemSize());
        for (int i = 0; i < img.rows; ++i) {
            const T* rowData = img.ptr<T>(i);
            memcpy(vecData.data() + i * colSize, rowData, colSize);
        }
    }
};


void readFileList(const std::string& basePath, std::vector<std::string>& imgFiles);
#endif //DEMO_UTIL_H
