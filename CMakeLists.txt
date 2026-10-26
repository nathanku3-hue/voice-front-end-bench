cmake_minimum_required(VERSION 3.16)
project(VoiceFrontEndBench VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_TESTING "Build Round 1 unit tests" ON)

# Use the installed FFTW3 numerical backend. On Ubuntu install libfftw3-dev;
# when only the runtime .so.3 is available find it explicitly.
find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3.so.3 fftw3.so.3 REQUIRED)

# cpu_isa.cpp is a separate translation unit so the CPUID probe runs once, at
# construction, and the hot path never touches it.
add_library(voice_fbank STATIC src/streaming_fbank.cpp src/cpu_isa.cpp)
target_include_directories(voice_fbank PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(voice_fbank PUBLIC "${FFTW3_LIBRARY}")
if(MSVC)
  target_compile_options(voice_fbank PRIVATE /W4)
else()
  target_compile_options(voice_fbank PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(voice_bounded_fbank STATIC src/bounded_fbank_bridge.cpp)
target_include_directories(voice_bounded_fbank PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(voice_bounded_fbank PUBLIC voice_fbank)

# Pure C++ greedy CTC decode + Byte-BPE detokenisation. No model runtime, no
# Python: the posteriors come from a frozen export of the same int8 model.
add_library(voice_ctc_decoder STATIC src/ctc_greedy_decoder.cpp)
target_include_directories(voice_ctc_decoder PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# AXI4-Stream protocol/timing model around the bounded bridge. Not RTL: it
# exists so the handshake under downstream backpressure can be dumped as a VCD.
add_library(voice_axi4s STATIC src/axi4s_adapter.cpp)
target_include_directories(voice_axi4s PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(voice_axi4s PUBLIC voice_bounded_fbank)

# Replaceable global operator new/delete. Linked ONLY into the allocation audit:
# it changes the allocation path of every translation unit it is linked with.
add_library(voice_alloc_probe STATIC src/alloc_probe.cpp)
target_include_directories(voice_alloc_probe PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(fbank_dump src/fbank_dump.cpp)
target_link_libraries(fbank_dump PRIVATE voice_fbank)

add_executable(fbank_bounded_dump src/fbank_bounded_dump.cpp)
target_link_libraries(fbank_bounded_dump PRIVATE voice_bounded_fbank)

# End-to-end C++ pipeline: WAV -> bounded Fbank -> greedy decode -> UTF-8 text.
add_executable(fbank_ctc_pipeline tools/fbank_ctc_pipeline.cpp)
target_include_directories(fbank_ctc_pipeline PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_link_libraries(fbank_ctc_pipeline PRIVATE voice_bounded_fbank voice_ctc_decoder)

# Raw feature dump (f32) plus the per-stage dynamic-range probe used by the
# fixed-point audit.
add_executable(fbank_feature_dump tools/fbank_feature_dump.cpp)
target_include_directories(fbank_feature_dump PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_link_libraries(fbank_feature_dump PRIVATE voice_fbank)

# AXI4-Stream simulation that writes the VCD rendered by scripts/render_vcd.py.
add_executable(axi4s_sim tools/axi4s_sim.cpp)
target_include_directories(axi4s_sim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_link_libraries(axi4s_sim PRIVATE voice_axi4s)

# Stage-wise latency budget for the frontend.
add_executable(fbank_stage_profile tools/fbank_stage_profile.cpp)
target_include_directories(fbank_stage_profile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_link_libraries(fbank_stage_profile PRIVATE voice_fbank)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_streaming tests/test_streaming.cpp)
  target_link_libraries(test_streaming PRIVATE voice_fbank)
  add_test(NAME streaming_invariance COMMAND test_streaming)

  add_executable(test_bounded_fbank_bridge tests/test_bounded_fbank_bridge.cpp)
  target_link_libraries(test_bounded_fbank_bridge PRIVATE voice_bounded_fbank)
  # The slow-consumer audit samples process RSS; Windows exposes that through
  # psapi rather than /proc/self/statm.
  if(WIN32)
    target_link_libraries(test_bounded_fbank_bridge PRIVATE psapi)
  endif()
  # Default soak is short so CTest stays fast. The recorded evidence run uses
  # --soak-seconds 3600 via scripts/run_backpressure_soak.py.
  add_test(NAME bounded_fbank_bridge COMMAND test_bounded_fbank_bridge)

  add_executable(test_ctc_greedy_decoder tests/test_ctc_greedy_decoder.cpp)
  target_link_libraries(test_ctc_greedy_decoder PRIVATE voice_ctc_decoder)
  add_test(NAME ctc_greedy_decoder COMMAND test_ctc_greedy_decoder)

  add_executable(test_zero_alloc_steady_state tests/test_zero_alloc_steady_state.cpp)
  target_link_libraries(test_zero_alloc_steady_state PRIVATE voice_bounded_fbank voice_alloc_probe)
  add_test(NAME zero_alloc_steady_state
           COMMAND test_zero_alloc_steady_state --frames 2000 --json "${CMAKE_CURRENT_BINARY_DIR}/zero_alloc.json")

  add_executable(test_hysteresis_flow tests/test_hysteresis_flow.cpp)
  target_link_libraries(test_hysteresis_flow PRIVATE voice_bounded_fbank)
  add_test(NAME hysteresis_flow
           COMMAND test_hysteresis_flow --sweep --frames 4000 --json "${CMAKE_CURRENT_BINARY_DIR}/hysteresis.json")

  add_executable(test_fast_log tests/test_fast_log.cpp)
  target_include_directories(test_fast_log PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_test(NAME fast_log_accuracy
           COMMAND test_fast_log --points 200000 --json "${CMAKE_CURRENT_BINARY_DIR}/fast_log.json")

  add_executable(test_axi4s_adapter tests/test_axi4s_adapter.cpp)
  target_link_libraries(test_axi4s_adapter PRIVATE voice_axi4s voice_alloc_probe)
  add_test(NAME axi4s_handshake
           COMMAND test_axi4s_adapter "${CMAKE_CURRENT_BINARY_DIR}/axi4s_protocol.json")

  add_executable(test_mel_sparsity tests/test_mel_sparsity.cpp)
  target_include_directories(test_mel_sparsity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")
  target_link_libraries(test_mel_sparsity PRIVATE voice_fbank)
  add_test(NAME mel_sparsity COMMAND test_mel_sparsity --seconds 5 --repeats 2)

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME bounded_fbank_cli
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bounded_fbank_cli.py"
        --baseline "$<TARGET_FILE:fbank_dump>" --bounded "$<TARGET_FILE:fbank_bounded_dump>")
    add_test(NAME pipeline_evidence
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_firmware_evidence.py")
  endif()
endif()
