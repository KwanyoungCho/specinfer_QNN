#include "qnn_qnnjson.h"

#include <fstream>
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

} // namespace llama_qnn


