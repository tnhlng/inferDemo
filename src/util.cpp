//
// Created by Admin on 2024/7/8.
//

#include "util.h"
using namespace std;




void readFileList(const string& basePath, vector<string>& imgFiles)
{
    const fs::path pathToTraverse{basePath}; // 替换为需要遍历的文件夹路径
    imgFiles.clear();
    for (const auto& entry : fs::recursive_directory_iterator(pathToTraverse)) {
        const auto& path = entry.path();
        if (entry.is_regular_file()) {
            imgFiles.push_back(entry.path().string());
        }
    }
}
