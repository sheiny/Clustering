set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
include_directories(${CMAKE_SOURCE_DIR})

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE .)

add_subdirectory(clustering)

add_executable( run_tests ${SOURCE} ${HEADERS} )
target_link_libraries(run_tests Catch src_clustering)

execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/test/input_files ${CMAKE_BINARY_DIR}/test/. RESULT_VARIABLE result )
