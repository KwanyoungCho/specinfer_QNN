#include "qnn_qnnjson.h"

#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>

#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace llama_qnn {

// Very lightweight parser tailored to qnn-context-binary-utility output.
// We scan by keys and rely on structure: info.graphs[].info.graphInputs/Outputs[].

static bool find_next(const std::string& s, size_t& pos, const std::string& key) {
  size_t p = s.find(key, pos);
  if (p == std::string::npos) return false;
  pos = p + key.size();
  return true;
}

static bool parse_string(const std::string& s, size_t& pos, std::string& out) {
  size_t q1 = s.find('"', pos);
  if (q1 == std::string::npos) return false;
  size_t q2 = s.find('"', q1 + 1);
  if (q2 == std::string::npos) return false;
  out = s.substr(q1 + 1, q2 - (q1 + 1));
  pos = q2 + 1;
  return true;
}

static bool parse_uint(const std::string& s, size_t& pos, uint64_t& v) {
  size_t p = s.find_first_of("0123456789", pos);
  if (p == std::string::npos) return false;
  size_t e = p;
  while (e < s.size() && (s[e] >= '0' && s[e] <= '9')) e++;
  v = std::stoull(s.substr(p, e - p));
  pos = e;
  return true;
}

static bool parse_float(const std::string& s, size_t& pos, double& out) {
  size_t p = s.find_first_of("-0123456789", pos);
  if (p == std::string::npos) return false;
  size_t e = p;
  while (e < s.size()) {
    char c = s[e];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
      e++;
      continue;
    }
    break;
  }
  try { out = std::stod(s.substr(p, e - p)); } catch (...) { return false; }
  pos = e;
  return true;
}

static uint32_t elem_size_from_dtype(const std::string& dt) {
  // 64-bit
  if (dt.find("INT_64") != std::string::npos || dt.find("UINT_64") != std::string::npos) return 8;
  // 32-bit
  if (dt.find("FLOAT_32") != std::string::npos || dt.find("BFLOAT_16_32") != std::string::npos || dt.find("UINT_32") != std::string::npos || dt.find("INT_32") != std::string::npos) return 4;
  // 16-bit (float16, bfloat16, fixed16, int16, uint16)
  if (dt.find("FLOAT_16") != std::string::npos || dt.find("BFLOAT_16") != std::string::npos || dt.find("UFIXED_POINT_16") != std::string::npos || dt.find("SFIXED_POINT_16") != std::string::npos || dt.find("UINT_16") != std::string::npos || dt.find("INT_16") != std::string::npos) return 2;
  // 8-bit (includes bool_8, fixed_point_8, int8/uint8)
  return 1;
}

static uint32_t elem_size_from_code(uint32_t code) {
  // map QNN type codes (see QnnTypes.h): 0x0232 float32, 0x0216 float16, 0x0132 u32, 0x0416 u16, 0x0408 u8, etc.
  switch (code) {
    case 0x0264: return 8; // float64
    case 0x0232: return 4; // float32
    case 0x0216: return 2; // float16
    case 0x0164: return 8; // uint64
    case 0x0132: return 4; // uint32
    case 0x0116: return 2; // uint16
    case 0x0108: return 1; // uint8
    case 0x0064: return 8; // int64
    case 0x0032: return 4; // int32
    case 0x0016: return 2; // int16
    case 0x0008: return 1; // int8
    case 0x0308: return 1; // sfix8
    case 0x0316: return 2; // sfix16
    case 0x0332: return 4; // sfix32
    case 0x0408: return 1; // ufix8
    case 0x0416: return 2; // ufix16
    case 0x0432: return 4; // ufix32
    case 0x0508: return 1; // bool8
    default: return 1;
  }
}

static uint32_t elem_size_from_qnn_dtype(Qnn_DataType_t data_type) {
  return elem_size_from_code(static_cast<uint32_t>(data_type));
}

static std::string qnn_dtype_to_string(Qnn_DataType_t data_type) {
  switch (data_type) {
    case QNN_DATATYPE_FLOAT_32: return "QNN_DATATYPE_FLOAT_32";
    case QNN_DATATYPE_FLOAT_16: return "QNN_DATATYPE_FLOAT_16";
    case QNN_DATATYPE_INT_8: return "QNN_DATATYPE_INT_8";
    case QNN_DATATYPE_INT_16: return "QNN_DATATYPE_INT_16";
    case QNN_DATATYPE_INT_32: return "QNN_DATATYPE_INT_32";
    case QNN_DATATYPE_INT_64: return "QNN_DATATYPE_INT_64";
    case QNN_DATATYPE_UINT_8: return "QNN_DATATYPE_UINT_8";
    case QNN_DATATYPE_UINT_16: return "QNN_DATATYPE_UINT_16";
    case QNN_DATATYPE_UINT_32: return "QNN_DATATYPE_UINT_32";
    case QNN_DATATYPE_UINT_64: return "QNN_DATATYPE_UINT_64";
    case QNN_DATATYPE_SFIXED_POINT_8: return "QNN_DATATYPE_SFIXED_POINT_8";
    case QNN_DATATYPE_SFIXED_POINT_16: return "QNN_DATATYPE_SFIXED_POINT_16";
    case QNN_DATATYPE_SFIXED_POINT_32: return "QNN_DATATYPE_SFIXED_POINT_32";
    case QNN_DATATYPE_UFIXED_POINT_8: return "QNN_DATATYPE_UFIXED_POINT_8";
    case QNN_DATATYPE_UFIXED_POINT_16: return "QNN_DATATYPE_UFIXED_POINT_16";
    case QNN_DATATYPE_UFIXED_POINT_32: return "QNN_DATATYPE_UFIXED_POINT_32";
    case QNN_DATATYPE_BOOL_8: return "QNN_DATATYPE_BOOL_8";
    default: return "QNN_DATATYPE_UNDEFINED";
  }
}

static std::string qnn_quant_encoding_to_string(Qnn_QuantizationEncoding_t encoding) {
  switch (encoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      return "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET";
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      return "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET";
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      return "QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET";
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
      return "QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET";
    case QNN_QUANTIZATION_ENCODING_BLOCK:
      return "QNN_QUANTIZATION_ENCODING_BLOCK";
    case QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION:
      return "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION";
    case QNN_QUANTIZATION_ENCODING_VECTOR:
      return "QNN_QUANTIZATION_ENCODING_VECTOR";
    default:
      return "QNN_QUANTIZATION_ENCODING_UNDEFINED";
  }
}

static bool tensor_desc_from_qnn_tensor(const Qnn_Tensor_t& tensor, QnnJsonTensorDesc& desc) {
  const Qnn_TensorV1_t* tensor_v1 = nullptr;
  const Qnn_TensorV2_t* tensor_v2 = nullptr;

  if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor_v2 = &tensor.v2;
  } else if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor_v1 = &tensor.v1;
  } else {
    return false;
  }

  const uint32_t id = tensor_v2 ? tensor_v2->id : tensor_v1->id;
  const char* name = tensor_v2 ? tensor_v2->name : tensor_v1->name;
  const Qnn_DataType_t data_type = tensor_v2 ? tensor_v2->dataType : tensor_v1->dataType;
  const Qnn_QuantizeParams_t& quant = tensor_v2 ? tensor_v2->quantizeParams : tensor_v1->quantizeParams;
  const uint32_t rank = tensor_v2 ? tensor_v2->rank : tensor_v1->rank;
  const uint32_t* dims = tensor_v2 ? tensor_v2->dimensions : tensor_v1->dimensions;

  desc = {};
  desc.id = id;
  desc.name = name ? name : "";
  desc.data_type_code = static_cast<uint32_t>(data_type);
  desc.data_type = qnn_dtype_to_string(data_type);
  desc.bytes_per_element = elem_size_from_qnn_dtype(data_type);

  if (dims != nullptr && rank > 0) {
    desc.dims.assign(dims, dims + rank);
  }

  uint64_t numel = 1;
  for (uint32_t dim : desc.dims) {
    numel *= dim;
  }
  desc.nbytes = numel * static_cast<uint64_t>(desc.bytes_per_element);

  desc.quant_encoding = qnn_quant_encoding_to_string(quant.quantizationEncoding);
  switch (quant.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      desc.quant_scale = quant.scaleOffsetEncoding.scale;
      desc.quant_offset = quant.scaleOffsetEncoding.offset;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      desc.quant_axis = quant.axisScaleOffsetEncoding.axis;
      if (quant.axisScaleOffsetEncoding.scaleOffset != nullptr) {
        for (uint32_t i = 0; i < quant.axisScaleOffsetEncoding.numScaleOffsets; ++i) {
          desc.quant_scales.push_back(quant.axisScaleOffsetEncoding.scaleOffset[i].scale);
          desc.quant_offsets.push_back(quant.axisScaleOffsetEncoding.scaleOffset[i].offset);
        }
      }
      break;
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      desc.quant_bitwidth = quant.bwScaleOffsetEncoding.bitwidth;
      desc.quant_scale = quant.bwScaleOffsetEncoding.scale;
      desc.quant_offset = quant.bwScaleOffsetEncoding.offset;
      break;
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
      desc.quant_bitwidth = quant.bwAxisScaleOffsetEncoding.bitwidth;
      desc.quant_axis = quant.bwAxisScaleOffsetEncoding.axis;
      if (quant.bwAxisScaleOffsetEncoding.scales != nullptr) {
        for (uint32_t i = 0; i < quant.bwAxisScaleOffsetEncoding.numElements; ++i) {
          desc.quant_scales.push_back(quant.bwAxisScaleOffsetEncoding.scales[i]);
          const int32_t offset =
              quant.bwAxisScaleOffsetEncoding.offsets != nullptr
                  ? quant.bwAxisScaleOffsetEncoding.offsets[i]
                  : 0;
          desc.quant_offsets.push_back(offset);
        }
      }
      break;
    default:
      break;
  }

  return !desc.name.empty();
}

template <typename GraphInfoT>
static void fill_graph_desc_from_system_info(
    const GraphInfoT& graph_info,
    std::map<std::string, QnnJsonGraphDesc>& out_graphs) {
  QnnJsonGraphDesc graph_desc;
  graph_desc.graph_name = graph_info.graphName ? graph_info.graphName : "";

  for (uint32_t i = 0; i < graph_info.numGraphInputs; ++i) {
    QnnJsonTensorDesc tensor_desc;
    if (tensor_desc_from_qnn_tensor(graph_info.graphInputs[i], tensor_desc)) {
      graph_desc.inputs.push_back(std::move(tensor_desc));
    }
  }

  for (uint32_t i = 0; i < graph_info.numGraphOutputs; ++i) {
    QnnJsonTensorDesc tensor_desc;
    if (tensor_desc_from_qnn_tensor(graph_info.graphOutputs[i], tensor_desc)) {
      graph_desc.outputs.push_back(std::move(tensor_desc));
    }
  }

  if (!graph_desc.graph_name.empty()) {
    out_graphs[graph_desc.graph_name] = std::move(graph_desc);
  }
}

static bool parse_dims(const std::string& s, size_t& pos, std::vector<uint32_t>& dims) {
  size_t lb = s.find('[', pos);
  size_t rb = s.find(']', pos);
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return false;
  size_t p = lb + 1;
  dims.clear();
  while (p < rb) {
    uint64_t v = 0;
    if (!parse_uint(s, p, v)) break;
    dims.push_back(static_cast<uint32_t>(v));
    size_t comma = s.find(',', p);
    if (comma == std::string::npos || comma > rb) break;
    p = comma + 1;
  }
  pos = rb + 1;
  return !dims.empty();
}

// ===== 안전한 범위 파싱용 헬퍼 추가 =====
static bool match_brace(const std::string& s, size_t lcurly, size_t& rcurly) {
  if (lcurly >= s.size() || s[lcurly] != '{') return false;
  int depth = 1;
  for (size_t i = lcurly + 1; i < s.size(); ++i) {
    if (s[i] == '{') depth++;
    else if (s[i] == '}') {
      depth--;
      if (depth == 0) { rcurly = i; return true; }
    }
  }
  return false;
}

static bool match_bracket(const std::string& s, size_t lbrack, size_t& rbrack) {
  if (lbrack >= s.size() || s[lbrack] != '[') return false;
  int depth = 1;
  for (size_t i = lbrack + 1; i < s.size(); ++i) {
    if (s[i] == '[') depth++;
    else if (s[i] == ']') {
      depth--;
      if (depth == 0) { rbrack = i; return true; }
    }
  }
  return false;
}

static bool parse_dims_block(const std::string& s, size_t& pos, std::vector<uint32_t>& dims) {
  size_t lb = s.find('[', pos);
  if (lb == std::string::npos) return false;
  int depth = 1;
  size_t i = lb + 1;
  dims.clear();
  uint64_t v = 0;
  bool in_num = false;
  while (i < s.size() && depth > 0) {
    char c = s[i];
    if (c == '[') {
      depth++;
    } else if (c == ']') {
      if (in_num) { dims.push_back(static_cast<uint32_t>(v)); v = 0; in_num = false; }
      depth--;
      if (depth == 0) { i++; break; }
    } else if (c >= '0' && c <= '9') {
      v = in_num ? (v * 10 + static_cast<uint64_t>(c - '0')) : static_cast<uint64_t>(c - '0');
      in_num = true;
    } else {
      if (in_num) { dims.push_back(static_cast<uint32_t>(v)); v = 0; in_num = false; }
    }
    i++;
  }
  pos = i;
  return !dims.empty();
}

static bool parse_tensor_object(const std::string& obj, QnnJsonTensorDesc& td) {
  size_t p = 0;
  // optional id
  size_t pid = 0;
  if (find_next(obj, pid, "\"id\"")) {
    if (find_next(obj, pid, ":")) {
      uint64_t idv = 0; if (parse_uint(obj, pid, idv)) td.id = static_cast<uint32_t>(idv);
    }
  }
  if (!find_next(obj, p, "\"name\"")) return false;
  if (!find_next(obj, p, ":")) return false;
  if (!parse_string(obj, p, td.name)) return false;
  p = 0;
  if (!find_next(obj, p, "\"dataType\"")) return false;
  if (!find_next(obj, p, ":")) return false;
  // dataType may be string or numeric code
  size_t backup = p;
  std::string dtype_str;
  if (parse_string(obj, p, dtype_str)) {
    td.data_type = dtype_str;
  } else {
    p = backup;
    uint64_t code=0; if (parse_uint(obj, p, code)) { td.data_type_code = static_cast<uint32_t>(code); }
  }
  // dimensions 키는 유틸리티 버전에 따라 currentDimensions로 나올 수 있음
  p = 0;
  bool ok = false;
  size_t save = p;
  if (find_next(obj, p, "\"currentDimensions\"")) {
    if (find_next(obj, p, ":") && parse_dims_block(obj, p, td.dims)) ok = true;
  }
  if (!ok) {
    p = save;
    if (find_next(obj, p, "\"dimensions\"")) {
      if (find_next(obj, p, ":") && parse_dims_block(obj, p, td.dims)) ok = true;
    }
  }
  if (!ok) return false;
  // quantization (best-effort)
  size_t pq = 0;
  if (find_next(obj, pq, "\"quantizeParams\"")) {
    // encoding
    size_t pe = pq;
    if (find_next(obj, pe, "\"quantizationEncoding\"") && find_next(obj, pe, ":")) {
      parse_string(obj, pe, td.quant_encoding);
    }
    // per-tensor scale/offset
    size_t ps = pq;
    if (find_next(obj, ps, "\"scale\"") && find_next(obj, ps, ":")) {
      double val = 0.0; if (parse_float(obj, ps, val)) td.quant_scale = static_cast<float>(val);
    }
    size_t po = pq;
    if (find_next(obj, po, "\"offset\"") && find_next(obj, po, ":")) {
      uint64_t iv = 0; if (parse_uint(obj, po, iv)) td.quant_offset = static_cast<int32_t>(iv);
    }
    // per-axis
    size_t pax = pq;
    if (find_next(obj, pax, "\"axis\"") && find_next(obj, pax, ":")) {
      uint64_t iv = 0; if (parse_uint(obj, pax, iv)) td.quant_axis = static_cast<int32_t>(iv);
    }
    size_t pss = pq;
    if (find_next(obj, pss, "\"scales\"") && find_next(obj, pss, "[")) {
      size_t lb = obj.find('[', pss); size_t rb = 0; if (match_bracket(obj, lb, rb)) {
        size_t cur = lb + 1;
        while (cur < rb) {
          double fv = 0.0; size_t tmp = cur; if (!parse_float(obj, tmp, fv)) break; td.quant_scales.push_back(static_cast<float>(fv));
          size_t comma = obj.find(',', tmp); if (comma == std::string::npos || comma > rb) { cur = rb; break; }
          cur = comma + 1;
        }
      }
    }
    size_t pofs = pq;
    if (find_next(obj, pofs, "\"offsets\"") && find_next(obj, pofs, "[")) {
      size_t lb = obj.find('[', pofs); size_t rb = 0; if (match_bracket(obj, lb, rb)) {
        size_t cur = lb + 1;
        while (cur < rb) {
          uint64_t iv = 0; size_t tmp = cur; if (!parse_uint(obj, tmp, iv)) break; td.quant_offsets.push_back(static_cast<int32_t>(iv));
          size_t comma = obj.find(',', tmp); if (comma == std::string::npos || comma > rb) { cur = rb; break; }
          cur = comma + 1;
        }
      }
    }
    size_t pbw = pq;
    if (find_next(obj, pbw, "\"bitwidth\"") && find_next(obj, pbw, ":")) {
      uint64_t iv = 0; if (parse_uint(obj, pbw, iv)) td.quant_bitwidth = static_cast<uint32_t>(iv);
    }
  }
  // bytesPerElement if present
  size_t pbe = 0;
  if (find_next(obj, pbe, "\"bytesPerElement\"") && find_next(obj, pbe, ":")) {
    uint64_t b=0; if (parse_uint(obj, pbe, b)) td.bytes_per_element = static_cast<uint32_t>(b);
  }
  uint64_t numel = 1; for (auto v : td.dims) numel *= v;
  uint32_t elem = td.bytes_per_element ? td.bytes_per_element : (td.data_type_code ? elem_size_from_code(td.data_type_code) : elem_size_from_dtype(td.data_type));
  if (!td.bytes_per_element) td.bytes_per_element = elem;
  td.nbytes = numel * static_cast<uint64_t>(elem);
  return true;
}

static void parse_tensor_array(const std::string& s, size_t array_key_pos, std::vector<QnnJsonTensorDesc>& out) {
  size_t lb = s.find('[', array_key_pos);
  if (lb == std::string::npos) return;
  size_t rb = 0; if (!match_bracket(s, lb, rb)) return;
  size_t cur = s.find('{', lb + 1);
  while (cur != std::string::npos && cur < rb) {
    size_t rc = 0; if (!match_brace(s, cur, rc) || rc > rb) break;
    std::string obj = s.substr(cur, rc - cur + 1);
    QnnJsonTensorDesc td;
    if (parse_tensor_object(obj, td)) out.push_back(std::move(td));
    cur = s.find('{', rc + 1);
  }
}

bool parse_qnn_json(const std::string& json_path,
                    std::map<std::string, QnnJsonGraphDesc>& out_graphs) {
  std::ifstream ifs(json_path);
  if (!ifs.is_open()) return false;
  std::stringstream buffer; buffer << ifs.rdbuf();
  std::string s = buffer.str();

  out_graphs.clear();

  size_t pos = 0;
  while (true) {
    size_t gkey = s.find("\"graphName\"", pos);
    if (gkey == std::string::npos) break;
    size_t lcurly = s.rfind('{', gkey);
    if (lcurly == std::string::npos) break;
    size_t rcurly = 0; if (!match_brace(s, lcurly, rcurly)) break;
    std::string block = s.substr(lcurly, rcurly - lcurly + 1);

    size_t p = 0;
    if (!find_next(block, p, "\"graphName\"")) { pos = rcurly + 1; continue; }
    if (!find_next(block, p, ":")) { pos = rcurly + 1; continue; }
    std::string gname; if (!parse_string(block, p, gname)) { pos = rcurly + 1; continue; }
    QnnJsonGraphDesc g; g.graph_name = gname;

    size_t inputs_key = block.find("\"graphInputs\"");
    size_t outputs_key = block.find("\"graphOutputs\"");
    if (inputs_key != std::string::npos) parse_tensor_array(block, inputs_key, g.inputs);
    if (outputs_key != std::string::npos) parse_tensor_array(block, outputs_key, g.outputs);

    out_graphs[g.graph_name] = std::move(g);
    pos = rcurly + 1;
  }
  return !out_graphs.empty();
}

bool parse_qnn_binary_info(void* system_so_handle,
                           const void* binary,
                           size_t binary_size,
                           std::map<std::string, QnnJsonGraphDesc>& out_graphs) {
  out_graphs.clear();

  if (system_so_handle == nullptr || binary == nullptr || binary_size == 0) {
    return false;
  }

  using GetProvidersFn = decltype(QnnSystemInterface_getProviders);
  using CreateFn = decltype(&QnnSystemContext_create);
  using GetBinaryInfoFn = decltype(&QnnSystemContext_getBinaryInfo);
  using GetMetadataFn = decltype(&QnnSystemContext_getMetadata);
  using FreeFn = decltype(&QnnSystemContext_free);

  auto get_providers_sym = dlsym(system_so_handle, "QnnSystemInterface_getProviders");

  CreateFn create_fn = nullptr;
  GetBinaryInfoFn get_binary_info_fn = nullptr;
  GetMetadataFn get_metadata_fn = nullptr;
  FreeFn free_fn = nullptr;

  if (get_providers_sym != nullptr) {
    auto get_providers = reinterpret_cast<GetProvidersFn*>(get_providers_sym);

    const QnnSystemInterface_t** providers = nullptr;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) == QNN_SUCCESS &&
        providers != nullptr &&
        num_providers > 0 &&
        providers[0] != nullptr) {
      const QnnSystemInterface_t* system_interface = providers[0];
      const auto& api = system_interface->QNN_SYSTEM_INTERFACE_VER_NAME;
      create_fn = api.systemContextCreate;
      get_binary_info_fn = api.systemContextGetBinaryInfo;
      get_metadata_fn = api.systemContextGetMetaData;
      free_fn = api.systemContextFree;
    }
  }

  if (create_fn == nullptr) {
    create_fn = reinterpret_cast<CreateFn>(dlsym(system_so_handle, "QnnSystemContext_create"));
  }
  if (get_binary_info_fn == nullptr) {
    get_binary_info_fn =
        reinterpret_cast<GetBinaryInfoFn>(dlsym(system_so_handle, "QnnSystemContext_getBinaryInfo"));
  }
  if (get_metadata_fn == nullptr) {
    get_metadata_fn =
        reinterpret_cast<GetMetadataFn>(dlsym(system_so_handle, "QnnSystemContext_getMetadata"));
  }
  if (free_fn == nullptr) {
    free_fn = reinterpret_cast<FreeFn>(dlsym(system_so_handle, "QnnSystemContext_free"));
  }

  if (create_fn == nullptr || free_fn == nullptr ||
      (get_metadata_fn == nullptr && get_binary_info_fn == nullptr)) {
    std::cerr << "[QNN BinaryInfo] Missing system metadata symbols, falling back to JSON\n";
    return false;
  }

  QnnSystemContext_Handle_t system_context = nullptr;
  const Qnn_ErrorHandle_t create_status = create_fn(&system_context);
  if (create_status != QNN_SUCCESS || system_context == nullptr) {
    std::cerr << "[QNN BinaryInfo] Failed to create QNN system context, status="
              << static_cast<unsigned long long>(create_status)
              << ", falling back to JSON\n";
    return false;
  }

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ErrorHandle_t info_status = QNN_SUCCESS;

  if (get_metadata_fn != nullptr) {
    info_status = get_metadata_fn(
        system_context,
        binary,
        static_cast<Qnn_ContextBinarySize_t>(binary_size),
        &binary_info);
  }

  if ((info_status != QNN_SUCCESS || binary_info == nullptr) && get_binary_info_fn != nullptr) {
    Qnn_ContextBinarySize_t binary_info_size = 0;
    info_status = get_binary_info_fn(
        system_context,
        const_cast<void*>(binary),
        static_cast<Qnn_ContextBinarySize_t>(binary_size),
        &binary_info,
        &binary_info_size);
  }

  if (info_status != QNN_SUCCESS || binary_info == nullptr) {
    free_fn(system_context);
    std::cerr << "[QNN BinaryInfo] Failed to query context metadata, status="
              << static_cast<unsigned long long>(info_status)
              << ", falling back to JSON\n";
    return false;
  }

  uint32_t num_graphs = 0;
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    num_graphs = binary_info->contextBinaryInfoV1.numGraphs;
    graphs = binary_info->contextBinaryInfoV1.graphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    num_graphs = binary_info->contextBinaryInfoV2.numGraphs;
    graphs = binary_info->contextBinaryInfoV2.graphs;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    num_graphs = binary_info->contextBinaryInfoV3.numGraphs;
    graphs = binary_info->contextBinaryInfoV3.graphs;
#endif
  } else {
    free_fn(system_context);
    std::cerr << "[QNN BinaryInfo] Unsupported binary info version "
              << static_cast<int>(binary_info->version)
              << ", falling back to JSON\n";
    return false;
  }

  if (graphs == nullptr || num_graphs == 0) {
    free_fn(system_context);
    std::cerr << "[QNN BinaryInfo] Binary metadata had no graphs, falling back to JSON\n";
    return false;
  }

  for (uint32_t i = 0; i < num_graphs; ++i) {
    if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      fill_graph_desc_from_system_info(graphs[i].graphInfoV1, out_graphs);
    } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
      fill_graph_desc_from_system_info(graphs[i].graphInfoV2, out_graphs);
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
    } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      fill_graph_desc_from_system_info(graphs[i].graphInfoV3, out_graphs);
#endif
    }
  }

  free_fn(system_context);
  return !out_graphs.empty();
}

} // namespace llama_qnn
