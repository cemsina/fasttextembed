cmake_minimum_required(VERSION 3.20)
project(fasttextembed_py C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-O3 -ffp-contract=off)

# SIMD features. NOTE for publishing: -mcpu=native targets the BUILD machine and is wrong for
# redistributable wheels. CI should override (e.g. -DCMAKE_C_FLAGS="-march=armv8.2-a+fp16" on
# arm64, or a baseline + future runtime dispatch). Fine for local installs.
include(CheckCCompilerFlag)
check_c_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
if(HAS_MCPU_NATIVE AND NOT DEFINED ENV{FTE_PORTABLE})
    add_compile_options(-mcpu=native)
endif()

# Repo layout when building in-tree; vendored ./csrc layout for sdist / isolated builds.
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
if(NOT EXISTS ${ROOT}/src)
    set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
endif()
file(GLOB CORE ${ROOT}/src/*.c ${ROOT}/src/kernels/*.c ${ROOT}/src/tokenizer/*.c)

add_library(fte SHARED ${CORE})
target_include_directories(fte PRIVATE ${ROOT}/include ${ROOT}/src)
target_link_libraries(fte m)

# Drop the compiled library into the importable package directory.
install(TARGETS fte LIBRARY DESTINATION fasttextembed RUNTIME DESTINATION fasttextembed)
