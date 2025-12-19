#include "qnn_tensor_util.h"

#include <cstring>

namespace llama_qnn {

static Qnn_TensorMemType_t kClient = QNN_TENSORMEMTYPE_RAW;

bool QnnTensorHolder::parse_data_type(const std::string& s, Qnn_DataType_t& out, uint32_t& bpe) {
  bpe = 1;
  if (s.find("FLOAT_32") != std::string::npos) { out = QNN_DATATYPE_FLOAT_32; bpe = 4; return true; }
  if (s.find("FLOAT_16") != std::string::npos) { out = QNN_DATATYPE_FLOAT_16; bpe = 2; return true; }
  // 일부 SDK에 BFLOAT_16 심볼이 없을 수 있으므로 생략
  if (s.find("INT_8") != std::string::npos) { out = QNN_DATATYPE_INT_8; bpe = 1; return true; }
  if (s.find("UINT_8") != std::string::npos) { out = QNN_DATATYPE_UINT_8; bpe = 1; return true; }
  if (s.find("INT_16") != std::string::npos) { out = QNN_DATATYPE_INT_16; bpe = 2; return true; }
  if (s.find("UINT_16") != std::string::npos) { out = QNN_DATATYPE_UINT_16; bpe = 2; return true; }
  if (s.find("INT_32") != std::string::npos) { out = QNN_DATATYPE_INT_32; bpe = 4; return true; }
  if (s.find("UINT_32") != std::string::npos) { out = QNN_DATATYPE_UINT_32; bpe = 4; return true; }
  if (s.find("INT_64") != std::string::npos) { out = QNN_DATATYPE_INT_64; bpe = 8; return true; }
  if (s.find("UINT_64") != std::string::npos) { out = QNN_DATATYPE_UINT_64; bpe = 8; return true; }
  // 기본값: 8-bit
  out = QNN_DATATYPE_UINT_8; bpe = 1; return true;
}

bool QnnTensorHolder::init_from_json(const QnnJsonTensorDesc& json_desc, void* data_ptr, uint64_t nbytes, bool is_input) {
  name_ = json_desc.name;
  dims_ = json_desc.dims;

  std::memset(&client_, 0, sizeof(client_));
  client_.data = data_ptr;
  client_.dataSize = static_cast<uint32_t>(nbytes);

  std::memset(&quant_, 0, sizeof(quant_));
  quant_.encodingDefinition = QNN_DEFINITION_DEFINED;
  quant_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

  std::memset(&tensor_, 0, sizeof(tensor_));
  tensor_.version = QNN_TENSOR_VERSION_2;
  tensor_.v2.id = json_desc.id; // ID 유지 시도
  tensor_.v2.name = name_.c_str();
  tensor_.v2.memType = kClient;
  tensor_.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor_.v2.type = is_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
  if (!dims_.empty()) {
    tensor_.v2.dimensions = dims_.data();
    tensor_.v2.rank = static_cast<uint32_t>(dims_.size());
  } else {
    tensor_.v2.dimensions = nullptr;
    tensor_.v2.rank = 0;
  }
  Qnn_DataType_t dt = QNN_DATATYPE_UINT_8; uint32_t bpe = 1;
  parse_data_type(json_desc.data_type, dt, bpe);
  tensor_.v2.dataType = dt;
  tensor_.v2.clientBuf = client_;
  // fill quant from json if available
  if (!json_desc.quant_encoding.empty()) {
    if (json_desc.quant_encoding.find("AXIS_SCALE_OFFSET") != std::string::npos) {
      quant_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
      quant_.axisScaleOffsetEncoding.axis = json_desc.quant_axis;
      quant_.axisScaleOffsetEncoding.numScaleOffsets = static_cast<uint32_t>(json_desc.quant_scales.size());
      // 주의: 여기서는 포인터를 외부 버퍼로 두지 않고, 정의만 표시(실사용 시 별도 버퍼 필요)
    } else if (json_desc.quant_encoding.find("SCALE_OFFSET") != std::string::npos) {
      quant_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      quant_.scaleOffsetEncoding.scale = json_desc.quant_scale;
      quant_.scaleOffsetEncoding.offset = json_desc.quant_offset;
    }
  }
  tensor_.v2.quantizeParams = quant_;
  tensor_.v2.isDynamicDimensions = NULL;
  return true;
}

void QnnTensorHolder::update_buffer(void* data_ptr, uint64_t nbytes) {
  // Update client buffer pointer and size
  client_.data = data_ptr;
  client_.dataSize = static_cast<uint32_t>(nbytes);
  
  // Update tensor reference to client buffer
  tensor_.v2.clientBuf = client_;
}

} // namespace llama_qnn


