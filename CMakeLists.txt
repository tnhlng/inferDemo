cmake_minimum_required(VERSION 3.15.3)

PROJECT(DEMO)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
message(${PROJECT_SOURCE_DIR})

set(OPENCV_ROOT "C:/thirdParty/opencv")
set(OpenCV_DIR "C:/thirdParty/opencv/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)

INCLUDE_DIRECTORIES("C:/thirdParty/opencv/include"
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/include/onnxruntime_cpu/
                    ${PROJECT_SOURCE_DIR}/include/json
)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib/onnxruntime_cpu/)
LINK_DIRECTORIES(C:/thirdParty/opencv/)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE( demo ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/facedetector.cpp
        include/util.h
        src/util.cpp
        include/application.h
        src/application.cpp)

target_link_libraries(demo onnxruntime ${OpenCV_LIBS})
