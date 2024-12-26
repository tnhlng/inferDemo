//
// Created by Admin on 2024/7/10.
//

#ifndef DEMO_APPLICATION_H
#define DEMO_APPLICATION_H

#endif //DEMO_APPLICATION_H
#include "facedetector.h"

class Application{
public:
    Application()=default;
    ~Application(){
        if(mFRec){
            delete mFRec;
        }
    }
    void runCamera();
    void runFolder(const string& path);
    void init();
private:
    FaceRec* mFRec = nullptr;
};