cmake_minimum_required(VERSION 3.16)
project(VoiceFrontEndBench VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_TESTING "Build Round 1 unit tests" ON)

# Use the installed FFTW3 numerical backend. On Ubuntu install libfftw3-dev;
# when only the runtime .so.3 is available find it explicitly.
find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3.so.3 fftw3.so.3 REQUIRED)

add_library(voice_fbank STATIC src/streaming_fbank.cpp)
target_include_directories(voice_fbank PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(voice_fbank PUBLIC "${FFTW3_LIBRARY}")
if(MSVC)
  target_compile_options(voice_fbank PRIVATE /W4)
else()
  target_compile_options(voice_fbank PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(fbank_dump src/fbank_dump.cpp)
target_link_libraries(fbank_dump PRIVATE voice_fbank)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_streaming tests/test_streaming.cpp)
  target_link_libraries(test_streaming PRIVATE voice_fbank)
  add_test(NAME streaming_invariance COMMAND test_streaming)
endif()
