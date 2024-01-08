cmake_minimum_required(VERSION 3.10)

project(rknn_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

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

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/src/main/native/include/librknnrt.so)

#rga
set(LIB_ARCH aarch64)
set(RGA_PATH ${CMAKE_SOURCE_DIR}/src/main/native/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

# opencv
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "lib")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME}
    src/main/native/cpp/rknn_jni.cpp
    src/main/native/cpp/rkYolov5s.cc
    src/main/native/cpp/preprocess.cc
    src/main/native/cpp/postprocess.cc
)

target_link_libraries(${PROJECT_NAME} 
    Threads::Threads
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
    ${RGA_LIB}
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/3rdparty/rga/RK3588/include
    ${JNI_INCLUDE_DIRS}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "\\\$ORIGIN/"
)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
install(FILES ${RKNN_RT_LIB} ${RGA_LIB} TYPE LIB)

add_executable(main_test src/main/native/cpp/main_test.cc)
target_link_libraries(main_test ${PROJECT_NAME})
target_include_directories(main_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
)
