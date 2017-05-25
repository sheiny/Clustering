set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
include_directories(${CMAKE_SOURCE_DIR})

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE .)

add_subdirectory(clustering)

if(CUDA_FOUND)
    add_subdirectory(clustering_gpu)
    cuda_add_executable( run_tests ${SOURCE} ${HEADERS} )
    target_link_libraries(run_tests Catch src_clustering)
else()
    add_executable( run_tests ${SOURCE} ${HEADERS} )
    target_link_libraries(run_tests Catch src_clustering)
endif()
