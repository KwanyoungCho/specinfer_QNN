#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llama_qnn {

// Minimal structures to carry graph I/O extracted from QNN SDK JSON
struct QnnJsonTensorDesc {
  uint32_t id {0};
  std::string name;
  std::vector<uint32_t> dims;
  std::string data_type; // e.g., "QNN_DATATYPE_UFIXED_POINT_8"
  uint32_t data_type_code {0}; // e.g., 1032 (0x0408), 1046 (0x0416), 562 (0x0232)
  uint32_t bytes_per_element {0};
  uint64_t nbytes {0};    // computed from dims * element size
  // 간단한 양자화 표현(필요 시 확장)
  std::string quant_encoding; // e.g., QNN_QUANTIZATION_ENCODING_SCALE_OFFSET / AXIS_SCALE_OFFSET
  float quant_scale {0.0f};
  int32_t quant_offset {0};
  int32_t quant_axis {0};
  std::vector<float> quant_scales;  // per-axis
  std::vector<int32_t> quant_offsets; // per-axis
  uint32_t quant_bitwidth {0};
};

struct QnnJsonGraphDesc {
  std::string graph_name; // e.g., prefill_forward / kv_forward
  std::vector<QnnJsonTensorDesc> inputs;
  std::vector<QnnJsonTensorDesc> outputs;
};

// QNN SDK JSON(forward_{i}_json.json)에서 그래프 I/O를 파싱
bool parse_qnn_json(const std::string& json_path,
                    std::map<std::string, QnnJsonGraphDesc>& out_graphs);

} // namespace llama_qnn


