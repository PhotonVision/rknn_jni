cmake_minimum_required(VERSION 3.10)

project(rknn_java)

add_subdirectory(utils)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(BUILD_SHARED_LIBS ON)

find_package(JNI)
if(JNI_FOUND)
    # Fixes odd AWT dependency
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME}
    src/main/native/cpp/rknn_yolo_wrapper.cc
    src/main/native/cpp/rknn_jni.cpp
    src/main/native/cpp/postprocess.cc
    src/main/native/cpp/yolov7.cc
)

target_link_libraries(${PROJECT_NAME} 
    Threads::Threads
    fileutils
    # ${CMAKE_CURRENT_SOURCE_DIR}/lib/librknnrt.so
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/rknpu2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${JNI_INCLUDE_DIRS}
)
