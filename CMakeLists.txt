cmake_minimum_required(VERSION 3.24)
project(gaudi2_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(SynapseAI REQUIRED)
find_package(Threads REQUIRED)

# Custom TPC kernels (GDN decode/verify, fused norm+quant, ...), loaded through GC_KERNEL_PATH at runtime.
add_subdirectory(tpc/flashinfer_gaudi)

add_library(g2e_core STATIC src/runtime.cpp src/graph.cpp src/weights.cpp src/ops.cpp src/qwen35.cpp src/gdn.cpp src/dspine.cpp src/generator.cpp src/vision.cpp src/image.cpp src/prefix_cache.cpp)
target_include_directories(g2e_core PUBLIC include)
target_include_directories(g2e_core SYSTEM PRIVATE third_party)
target_link_libraries(g2e_core PUBLIC SynapseAI::SynapseAI Threads::Threads)
target_compile_options(g2e_core PRIVATE -Wall -Wextra)
target_compile_definitions(g2e_core PRIVATE G2E_TPC_LIB="$<TARGET_FILE:flashinfer_gaudi_kernels>")
add_dependencies(g2e_core flashinfer_gaudi_kernels)

# ---- text: tokenizer + chat template (vendored llama.cpp unicode/json/jinja, MIT)
file(GLOB G2E_JINJA_SOURCES third_party/llama/jinja/*.cpp)
add_library(g2e_text STATIC src/tokenizer.cpp third_party/llama/unicode.cpp third_party/llama/unicode-data.cpp
            third_party/llama/json.cpp third_party/llama/common_unicode.cpp ${G2E_JINJA_SOURCES})
target_include_directories(g2e_text PUBLIC include third_party PRIVATE third_party/llama)

# ---- OpenAI-compatible server
add_executable(g2e_server apps/g2e_server.cpp src/qwen3_parser.cpp third_party/httplib/httplib.cpp)
target_include_directories(g2e_server PRIVATE third_party)
target_link_libraries(g2e_server PRIVATE g2e_core g2e_text)
