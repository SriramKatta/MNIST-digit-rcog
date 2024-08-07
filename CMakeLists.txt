cmake_minimum_required(VERSION 3.14.4)

project(Ministdataset)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(COPY ${CMAKE_SOURCE_DIR}/mnist-datasets DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_SOURCE_DIR}/mnist-configs DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

include(FetchContent)

include(external/eigen3.cmake)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(implementation)
