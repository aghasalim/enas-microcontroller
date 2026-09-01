# Host build of the reference implementation and its equivalence test.
# The Makefile next to this file does the same job with less ceremony; this
# exists so the C and C++ both get built by a second, stricter toolchain in CI.
cmake_minimum_required(VERSION 3.16)
project(micronet C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic -O2)

add_library(micronet STATIC micronet.c)
target_include_directories(micronet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_equivalence test_equivalence.c)
target_link_libraries(test_equivalence micronet m)

add_executable(bench bench.cpp)
target_link_libraries(bench micronet)

enable_testing()
add_test(NAME equivalence
         COMMAND test_equivalence ${CMAKE_CURRENT_SOURCE_DIR}/generated)
