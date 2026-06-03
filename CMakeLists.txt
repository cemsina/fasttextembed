cmake_minimum_required(VERSION 3.20)
project(fasttextembed C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-O3 -ffp-contract=off -Wall -Wextra)

# Enable host-CPU SIMD features (fp16 FMA on Apple Silicon/ARM; AVX2+FMA+F16C on x86).
# Pick the flag by architecture: on x86, GCC ACCEPTS -mcpu=native but it does NOT enable
# AVX2 (it is a deprecated -mtune alias), so x86 must use -march=native.
include(CheckCCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[0-9]86")
    check_c_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
else()
    check_c_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)   # ARM (Apple clang + GCC)
    check_c_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MCPU_NATIVE)
        add_compile_options(-mcpu=native)
    elseif(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

include_directories(include src)

file(GLOB CORE
    src/*.c
    src/kernels/*.c
    src/tokenizer/*.c)
add_library(fte STATIC ${CORE})
set_property(TARGET fte PROPERTY POSITION_INDEPENDENT_CODE ON)

# Shared library for language bindings (Python ctypes, Node N-API). Same sources.
option(FTE_SHARED "Build the shared library libfte" ON)
if(FTE_SHARED)
    add_library(fte_shared SHARED ${CORE})
    set_target_properties(fte_shared PROPERTIES OUTPUT_NAME fte)
    target_link_libraries(fte_shared m)
endif()

add_executable(fte_cli cli/main.c)
target_link_libraries(fte_cli fte m)

enable_testing()
foreach(t test_kernels test_tokenizer test_e2e)
    add_executable(${t} tests/${t}.c)
    target_link_libraries(${t} fte m)
    add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
