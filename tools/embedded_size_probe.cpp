#include "streaming_fbank.h"
#include "bounded_fbank_bridge.h"
#include "ctc_greedy_decoder.h"
#include "axi4s_adapter.h"
#include "fast_log.h"
#include "flow_hysteresis.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  std::printf("POINTER_SIZE: %zu\n", sizeof(void*));
  std::printf("SIZEOF_STREAMING_FBANK: %zu\n", sizeof(voice::StreamingFbank));
  std::printf("SIZEOF_BOUNDED_BRIDGE: %zu\n", sizeof(voice::BoundedFbankBridge));
  std::printf("SIZEOF_CTC_DECODER: %zu\n", sizeof(voice::CtcGreedyDecoder));
  std::printf("SIZEOF_AXI4S_ADAPTER: %zu\n", sizeof(voice::Axi4sAdapter));
  std::printf("SIZEOF_FLOW_HYSTERESIS: %zu\n", sizeof(voice::HysteresisGate));
  std::printf("SIZEOF_MEL_TAP: %zu\n", sizeof(voice::StreamingFbank::MelTap));
  return 0;
}
