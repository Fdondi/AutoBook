cmake_minimum_required(VERSION 3.22.1)
project(autoreader)

# llama.cpp, pinned to a release tarball and its SHA-256: bump both together.
# CPU only. Static libraries, so the app ships one libautoreader.so.
include(FetchContent)
FetchContent_Declare(llama_cpp
        URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11450.tar.gz
        URL_HASH SHA256=bc717d30da4d3c0546aded0f254349fd272a65fd1c87638ab06b98f781d7a54b)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)           # chat helpers, JSON schema -> grammar
foreach(opt LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP
            LLAMA_OPENSSL LLAMA_SUBPROCESS GGML_OPENMP GGML_NATIVE)
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
if(ANDROID_ABI STREQUAL "arm64-v8a")
    # dotprod is on every recent phone; KleidiAI picks faster kernels (i8mm, SME) at runtime.
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
elseif(ANDROID_ABI STREQUAL "x86_64")                    # emulators: the host CPU has AVX2
    foreach(opt GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C)
        set(${opt} ON CACHE BOOL "" FORCE)
    endforeach()
endif()
FetchContent_MakeAvailable(llama_cpp)

add_library(autoreader SHARED engine.cpp)
target_link_libraries(autoreader llama llama-common android log)
