project(clustering)
cmake_minimum_required(VERSION 3.2.2)
set (CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/3rdparty/boost_1_65_1/)

add_subdirectory(src)
add_subdirectory(test)
