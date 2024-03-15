cmake_minimum_required(VERSION 3.10)

project(rknn_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
endif ()

set(BUILD_SHARED_LIBS ON)

find_package(JNI)
if(JNI_FOUND)
    # Fixes odd AWT dependency
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/src/main/native/lib/librknnrt.so)

#rga
set(LIB_ARCH aarch64)
set(RGA_PATH ${CMAKE_SOURCE_DIR}/src/main/native/lib/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)


set(OPENCV_YEAR "frc2024")
set(OPENCV_VERSION "4.8.0-2")
# also valid: windowsx86-64

# type can be "", "debug", "static", or "staticdebug"
set(OPENCV_TYPE "")

# Download opencv, and save the path
include(FetchContent)
FetchContent_Declare(
    opencv_lib
    URL https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
)
FetchContent_MakeAvailable(opencv_lib)

# download OpenCV headers
FetchContent_Declare(
    opencv_header
    URL https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
)
FetchContent_MakeAvailable(opencv_header)

# This probably doesn't work great with shared libraries, but I don't care about those right now
file(GLOB_RECURSE OPENCV_LIB_PATH
  "${opencv_lib_SOURCE_DIR}/**/*.lib"
  "${opencv_lib_SOURCE_DIR}/**/*.so*"
)
SET(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on opencv ${OPENCV_LIB_PATH}")


set(CMAKE_INSTALL_RPATH "lib")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME}
    src/main/native/cpp/rknn_jni.cpp
    src/main/native/cpp/yolo_common.cpp
    src/main/native/cpp/preprocess.cc
    src/main/native/cpp/postprocess_v5.cc
    src/main/native/cpp/postprocess_v8.cc
)


target_link_libraries(${PROJECT_NAME} PUBLIC
    Threads::Threads
    ${RKNN_RT_LIB}
    ${OPENCV_LIB_PATH}
    ${RGA_LIB}
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/model_zoo
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/3rdparty/rga/RK3588/include
    ${JNI_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_PATH}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "\\\$ORIGIN/"
)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
install(FILES ${RKNN_RT_LIB} ${RGA_LIB} TYPE LIB)

message("RGA is ${RGA_LIB}")

add_executable(main_test src/main/native/cpp/main_test.cc)
target_link_libraries(main_test ${PROJECT_NAME} ${RGA_LIB})
target_include_directories(main_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/model_zoo
    ${OPENCV_INCLUDE_PATH}
)
