#include "facedetector.h"
#include "util.h"
#include <cmath>
#include <fstream>
#include <set>
#include <format>

FaceRec::FaceRec(std::vector<string> model_dir)
{
    m_model_dir.assign(model_dir.begin(), model_dir.end());
    cout << "m_model_dir :" << m_model_dir.size() << endl;
}

FaceRec::~FaceRec()
{
    session_FaceNet.reset();
    m_OrtEnv.reset();
}

void FaceRec::Init()
{
    cout << "Init()......" << endl;
    m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    session_option.SetIntraOpNumThreads(1);
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    if (0 != GetOnnxModelInfo(m_model_dir))
    {
        cout<<"ModelInit failed!!!"<<endl;
        return;
    }
    dist_threshold = 0.65;
    this->loadFaceBase();
}


void FaceRec::GetOnnxModelInputInfo(const Ort::Session &session_net,
                                            std::vector<const char*> &input_node_names,
                                            vector<vector<int64_t> > &input_node_dims,
                                            std::vector<const char*> &output_node_names,
                                            vector<vector<int64_t> > &output_node_dims)
{
    size_t num_input_nodes = session_net.GetInputCount();
    input_node_names.resize(num_input_nodes);

    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++)
    {
        char* input_name = session_net.GetInputName(i, allocator);
        input_node_names[i] = input_name;

        Ort::TypeInfo type_info = session_net.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();

        std::vector<int64_t> inputNodeDims = tensor_info.GetShape();
        input_node_dims.push_back(inputNodeDims);
    }

    size_t num_output_nodes = session_net.GetOutputCount();
    output_node_names.resize(num_output_nodes);
    //std::vector<int64_t> output_node_dims;
    //char* output_name = nullptr;

    for (int i = 0; i < num_output_nodes; i++)
    {
        char* output_name = session_net.GetOutputName(i, allocator);
        output_node_names[i] = output_name;

        Ort::TypeInfo type_info = session_net.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
        output_node_dims.push_back(outputNodeDims);
    }

}

int64_t FaceRec::GetOnnxModelInfo(std::vector<string> model_dir)
{
    if (model_dir.empty())
    {
        cout << "model_dir empty, please check it!!!" << endl;
        return -1;
    }
    m_faceModel_dir = model_dir[0];
    auto fileFace      = transWcharPath(m_faceModel_dir);

    session_FaceNet = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv,fileFace.c_str(),session_option));
    GetOnnxModelInputInfo(*session_FaceNet,m_FaceNetInputNodeNames,m_FaceNetInputNodesDims,m_FaceNetOutputNodeNames,m_FaceNetOutputNodesDims);

    return 0;
}

//#define IMAGE_DEBUG
vector< FaceRec::FaceInfo > FaceRec::Detect(const cv::Mat& img)
{
    auto faceId = this->faceExtract({img});
    std::vector<FaceInfo> faces;
    for(const auto& id : faceId){
        FaceInfo fi;
        fi.emb = id;
        fi.faceName = this->who(id);
        faces.push_back(fi);
    }
    return faces;
}


float FaceRec::faceIdCmp(const vector<float> &face1, const vector<float> &face2) {
    if(face1.size() != face2.size()){
        return INT_MAX;
    }
    float sum = 0.0;
    for(int i = 0;i<face1.size();i++){
        sum += pow((face1[i] - face2[i]),2.0);
    }
    return sqrt(sum);
}


void FaceRec::loadFaceBase() {
    std::filesystem::path current_path = std::filesystem::current_path();
    //加载已有人脸库
    const string faceBaseFile = current_path.string() + "/../data/emb/faceset.json";
    std::stringstream ss;
    std::ifstream file(faceBaseFile);
    try {
        if (file.is_open()) {
            std::string file_contents{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
            ss.str(file_contents);
            ss >> mFaceSet;
        }
    }
    catch(std::exception& e){
        mFaceSet = json();
    }
    set<string> fileSet;
    //刷新、加载新人脸。
    for(const auto& item : mFaceSet.items()) {
        auto name = item.key();
        auto faceId = item.value();
        for (const auto &emb_base: faceId) {
            auto fileName = emb_base["file"].get<string>();
            auto id = emb_base["value"].get<vector<float>>();
            fileSet.insert(fileName);
        }
    }
    const fs::path pathToTraverse =  current_path.string() + "/../data/faceset/";
    for (const auto& entry : fs::recursive_directory_iterator(pathToTraverse)) {
        const auto& path = entry.path();
        if(!entry.is_directory()) {
            continue;
        }
        auto faceName = path.filename().string();
        const fs::path secPath = pathToTraverse.string() + "/" + faceName;
        for(const auto& secEntry : fs::recursive_directory_iterator(secPath)){
            if (!secEntry.is_regular_file()) {
                continue;
            }
            auto tAbsFileName = secEntry.path().string();
            auto tFileName = secEntry.path().filename().string();
            if(fileSet.find(tFileName) == fileSet.end()){
                auto img =  cv::imread(tAbsFileName);
                auto boxes = this->Detect(img);
                if(boxes.empty()){
                    continue;
                }

                if(boxes.front().emb.empty()){
                    continue;
                }
                boxes.front().faceName = faceName;
                if(mFaceSet.find(faceName) == mFaceSet.end()){
                    mFaceSet[faceName] = {};
                }
                json sj;
                sj["name"] = faceName;
                sj["file"] = tFileName;
                sj["value"] = boxes.front().emb;
                mFaceSet[faceName].push_back(sj);
            }
        }
    }
    std::ofstream os(faceBaseFile,'w');
    os << mFaceSet;
    os.close();
}

vector<vector<float>> FaceRec::faceExtract(const vector<cv::Mat> &imgs) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    int batch = imgs.size();
    int channel = 3;
    int height_r = 224;
    int width_r = 224;

    std::array<int64_t, 4> input_shape_{ batch, channel, height_r, width_r };

    vector<int64_t>  output_node_dims_0;
    vector<vector<float>> batchFaceId;

    size_t input_tensor_size = batch * channel * height_r * width_r;
    std::vector<float> input_image_(input_tensor_size);
    for (int i = 0; i < batch; i++) {
        cv::Mat tImg,cImg,rImg;
        cv::resize(imgs[i],tImg,Size(224,224));

        cv::cvtColor(tImg,cImg,COLOR_BGR2RGB);
        tImg.convertTo(rImg,CV_32FC3,1.0/255.0);

        int bytesSize = channel * height_r * width_r;
        int beginIndex = i * bytesSize;
        float *input_data = input_image_.data();
        fill(input_image_.begin() + beginIndex, input_image_.begin() + (i + 1) * bytesSize, 0.f);

        vector<float> tmp;
        imageChannelCopyTo(rImg,tmp);
        memcpy(input_data+ beginIndex,tmp.data(),tmp.size()*sizeof(float));
    }
    if(true){
        // create input tensor object from data values
        Ort::Value input_tensor_facenet = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

        auto output_tensors_facenet = session_FaceNet->Run(Ort::RunOptions{nullptr}, m_FaceNetInputNodeNames.data(), &input_tensor_facenet, m_FaceNetInputNodeNames.size(), m_FaceNetOutputNodeNames.data(), m_FaceNetOutputNodeNames.size());

        //conv5-2_Gemm_Y
        Ort::TypeInfo type_info_0 = output_tensors_facenet[0].GetTypeInfo();
        auto tensor_info_0 = type_info_0.GetTensorTypeAndShapeInfo();
        size_t tensor_size_0 = tensor_info_0.GetElementCount();
        output_node_dims_0 = tensor_info_0.GetShape();
        batchFaceId.clear();
        batchFaceId.resize(output_node_dims_0.front());
        auto faceIdSize = output_node_dims_0[1];
        float *outarr0 = output_tensors_facenet[0].GetTensorMutableData<float>();

        for (int j = 0; j < tensor_size_0; j++)
        {
            auto v = outarr0[j];
            auto index = int(j / faceIdSize);
            if(batchFaceId[index].capacity() < faceIdSize){
                batchFaceId[index].reserve(faceIdSize);
            }
            batchFaceId[index].push_back(v);
        }

    }

    return batchFaceId;
}

string FaceRec::who(const vector<float> &emb) {
    float minDist = 99.0;
    string name_res = "unknown";
    for(const auto& item : mFaceSet.items()){
        auto name = item.key();
        auto faceId = item.value();
        for(const auto& emb_base : faceId){
            auto fileName = emb_base["file"].get<string>();
            auto id = emb_base["value"].get<vector<float>>();
            float dist = faceIdCmp(emb,id);
            if(dist < this->dist_threshold){
                if(dist < minDist){
                    name_res = name;
                    minDist = dist;
                }
            }
        }
    }
    return name_res;
}