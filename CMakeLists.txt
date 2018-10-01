cmake_minimum_required(VERSION 2.8)
project(DLab)

set(SRCS 
  src/dlab.cpp     
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(DLab SHARED ${SRCS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(DLab ${OpenCV_LIBS})
