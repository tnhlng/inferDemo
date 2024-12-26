#ifndef _FACE_DETECTOR_H_
#define _FACE_DETECTOR_H_
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <locale>
#include <codecvt>

using namespace std;
using namespace cv;

inline std::wstring to_wide_string(const std::string& input) //string to wstring
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(input);
}

#ifdef _WIN32
#include "onnxruntime_cxx_api.h"
#include "util.h"

typedef ORTCHAR_T ONNX_PATH_CH_T;

inline const wstring transWcharPath(const std::string& path) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wide_str = converter.from_bytes(path);
    return wide_str;
}

#else
typedef char ONNX_PATH_CH_T;
inline const ONNX_PATH_CH_T* transWcharPath(const std::string& path) {
	return path.c_str();
}
#endif

class FaceRec
{
public:

    struct FaceInfo{
        string faceName;
        vector<float> emb;
    };

private:
    float                            dist_threshold;

private:
    Ort::SessionOptions session_option;
    std::unique_ptr<Ort::Session>    session_FaceNet;
    std::unique_ptr<Ort::Env>        m_OrtEnv;

    std::vector<string> m_model_dir;
    std::string      m_faceModel_dir;

    vector<const char*>         m_FaceNetInputNodeNames;
    vector<vector<int64_t>>     m_FaceNetInputNodesDims;

    vector<const char*>         m_FaceNetOutputNodeNames;
    vector<vector<int64_t>>     m_FaceNetOutputNodesDims;

    std::mutex m_onnx_mutex;
private:
    json mFaceSet;

public:
    FaceRec(std::vector<string> model_dir);

    ~FaceRec();
    vector< FaceInfo > Detect (const cv::Mat& img);

    void Init();
    void GetOnnxModelInputInfo(const Ort::Session &session_net,
                                        std::vector<const char*> &input_node_names,
                                        vector<vector<int64_t> > &input_node_dims,
                                        std::vector<const char*> &output_node_names,
                                        vector<vector<int64_t> > &output_node_dims);
    int64_t GetOnnxModelInfo(std::vector<string> model_dir);
private:

    float faceIdCmp(const vector<float>& face1,const vector<float>& face2);

    void loadFaceBase();//加载数据库中已有人脸。


    vector<vector<float>> faceExtract(const vector<cv::Mat> &imgs);

    string who(const vector<float>& emb);

};

#endif