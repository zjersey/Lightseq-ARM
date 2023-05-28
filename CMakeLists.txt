cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(Protobuf_USE_STATIC_LIBS ON)

project(LightSeq-CPU)

add_definitions(-w)
set(Protobuf_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/protobuf-3.21.4/src)
include_directories(${Protobuf_INCLUDE_DIRS})

option(DEBUG_MODE "debug computation result" OFF)
option(PROFILE "profiling mode" OFF)
option(BUILD_DYNAMIC "build dynamic libs" OFF)

if(DEBUG_MODE)
  add_definitions(-DDEBUG)
  message(STATUS "Debug computation result")
endif()

if(PROFILE)
  add_definitions(-DPROFILE)
  message(STATUS "Profiling mode")
endif()

set(CMAKE_CXX_FLAGS "-march=armv8-a+dotprod")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -frtti -fopenmp")
set(CMAKE_C_FLAGS "-O3 -g -frtti -fopenmp")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-openmp")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-openmp")
endif()

add_subdirectory(src/models)
add_subdirectory(src/proto)
add_subdirectory(src/ops)
add_subdirectory(src/ops/chgemm)
add_subdirectory(src/utils)

add_executable(demo "demo.cpp")
target_link_libraries(demo lightseq_model)