//
// Created by Admin on 2024/7/10.
//

#include "application.h"

void Application::init() {
    std::filesystem::path current_path = std::filesystem::current_path();
    const string faceFile = current_path.string() + "/../model/FaceNet.onnx";
    vector<string> modelDir = {faceFile};
    mFRec = new FaceRec(modelDir);
    mFRec->Init();
}

void Application::runCamera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: 摄像头未能打开!" << std::endl;
        return ;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: 帧为空!" << std::endl;
            break;
        }
        vector<FaceRec::FaceInfo> res = mFRec->Detect(frame);
    }

    cap.release();
    cv::destroyAllWindows();
}

void Application::runFolder(const std::string &path) {
    vector<string> imgList;
    readFileList(path,imgList);
    cout << "imgList size : " << imgList.size() << endl;
    for(const auto& imgPath : imgList)
    {
        cv::Mat testImg = cv::imread(imgPath);
        if(testImg.empty()){
            cout<<"file: "<<imgPath << ",empty img"<<endl;
            continue;
        }
        vector<FaceRec::FaceInfo> res = mFRec->Detect(testImg);
        cout<<"file: "<< imgPath << ", face id cnt: "<< res.size()<<endl;
        for(auto idx = 0 ; idx < res.size() ; idx ++){
            cout<< "\tface idx: " << idx << ", name: " << res[idx].faceName << endl;
        }
    }
}
