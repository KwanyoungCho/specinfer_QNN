#pragma once

#include "qnn_qnnjson.h"

#include <QnnTypes.h>
#include <QnnTensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace llama_qnn {

// QNN 텐서 홀더: 수명 관리(이름/차원/클라이언트 버퍼)를 캡슐화하여
// Qnn_Tensor_t 내부 포인터가 안전하게 유지되도록 한다.
class QnnTensorHolder {
public:
  QnnTensorHolder() = default;

  // JSON 텐서 메타와 호스트 메모리 포인터로 Qnn_Tensor_t를 구성한다.
  // - data_ptr: QNNIOAllocator에서 받은 텐서 버퍼 주소
  // - nbytes: 해당 버퍼의 바이트 수
  // - is_input: 입력(true) / 출력(false)
  // 반환값: 성공 여부
  bool init_from_json(const QnnJsonTensorDesc& json_desc, void* data_ptr, uint64_t nbytes, bool is_input);

  // Update buffer pointer without reconstructing the whole tensor
  void update_buffer(void* data_ptr, uint64_t nbytes);

  const Qnn_Tensor_t& tensor() const { return tensor_; }

private:
  static bool parse_data_type(const std::string& s, Qnn_DataType_t& out, uint32_t& bpe);

  std::string name_;
  std::vector<uint32_t> dims_;
  Qnn_ClientBuffer_t client_ { };
  Qnn_QuantizeParams_t quant_ { };
  Qnn_Tensor_t tensor_ { };
};

} // namespace llama_qnn


