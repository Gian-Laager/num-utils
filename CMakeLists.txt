cmake_minimum_required(VERSION 3.20)

project(num-utils)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_DEBUG_POSTFIX _d)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

option(BUILD_TESTS "Create target to build tests" ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Relase")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
endif()

include(FetchContent)

if(TARGET Eigen3::Eigen)
    message(STATUS "using existign Eigen3::Eigen")
else ()
    FetchContent_Declare(
      Eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG master
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE)

    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_USE_LAPACKE ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    FetchContent_MakeAvailable(Eigen)
endif()

file(GLOB SRC "src/**.cpp")
file(GLOB HEAD "include/**.h")
list(FILTER HEAD EXCLUDE REGEX ".*pch.h$")

set(LIB_NAME "${PROJECT_NAME}")

add_library(${LIB_NAME} STATIC)
target_sources(${LIB_NAME}
    PRIVATE
        ${SRC}
    PUBLIC
        ${HEAD}
)
target_link_libraries(${LIB_NAME} Eigen3::Eigen)
target_include_directories(${LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_precompile_headers(${LIB_NAME} PUBLIC include/nu/pch.h)

add_library(NumUtils::nu ALIAS ${LIB_NAME})

if (BUILD_TESTS)
    add_subdirectory(test)
endif()
