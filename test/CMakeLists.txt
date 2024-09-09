include(FetchContent)

FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG  release-1.11.0
)

FetchContent_MakeAvailable(googletest)


file(GLOB_RECURSE TEST_SRC "*.cpp")
file(GLOB_RECURSE TEST_HEAD "*.h")
# list(FILTER TEST_HEAD EXCLUDE REGEX ".*pch.h$")

set(TESTS_NAME "${PROJECT_NAME}-test")

add_executable(${TESTS_NAME} ${TEST_SRC} ${TEST_HEAD})

# target_precompile_headers(${TESTS_NAME} PRIVATE "pch.h")
target_link_libraries(${TESTS_NAME} PRIVATE gtest ${LIB_NAME})
target_include_directories(${TESTS_NAME} PRIVATE gtest ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_NAME})
