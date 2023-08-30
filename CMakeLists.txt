cmake_minimum_required(VERSION 3.26)
project(EXAMEN_FINAL)

set(CMAKE_CXX_STANDARD 17)

#### FMT
include(FetchContent)

FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

add_executable(EXAMEN_FINAL main.cpp)

target_link_libraries(EXAMEN_FINAL fmt::fmt)