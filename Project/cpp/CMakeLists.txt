cmake_minimum_required(VERSION 3.14)
project(nids LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(nids nids.cpp)
target_link_libraries(nids PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(nids PRIVATE -O2 -Wall -Wextra)
