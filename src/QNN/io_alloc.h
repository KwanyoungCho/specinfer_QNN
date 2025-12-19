#pragma once

#include "qnn_qnnjson.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace llama_qnn {

// ExecuTorch 흐름을 최대한 그대로 모사한 I/O 텐서 할당기(간소화 버전)
// - 현재 단계에서는 mutable buffer(id)가 모두 -1이므로, 텐서별(per-tensor) 할당만 수행
// - 추후 MethodMeta 기반 mutable buffer 플랜을 받으면 동일 인터페이스에서 확장 가능
class QNNIOAllocator {
public:
  // 그래프 I/O 메타(이 단계에서는 QNN SDK JSON에서 파싱한 결과)를 입력으로 받아
  // 내부 상태(텐서명 -> nbytes)를 구성한다.
  void build_from_qnnjson(const QnnJsonGraphDesc& desc);

  // 실제 메모리 할당을 수행한다.
  // - alignment는 2의 거듭제곱 권장(64 등)
  // - 이미 할당된 메모리가 있으면 release() 후 새로 할당
  // - 반환값은 총 할당 바이트
  std::uint64_t allocate(std::size_t alignment);

  // 텐서명 -> 할당된 버퍼 주소 바인딩(그래프 바인딩 시 clientBuf.data로 사용)
  const std::map<std::string, void*>& bindings() const { return name_to_ptr_; }

  // 총 할당 바이트(마지막 allocate 기준)
  std::uint64_t total_allocated_bytes() const { return total_allocated_bytes_; }

  // 보유 중인 모든 버퍼 해제(멱등)
  void release();

private:
  static bool is_pow2(std::size_t v) { return v && ((v & (v - 1)) == 0); }

  std::map<std::string, std::uint64_t> name_to_nbytes_;
  std::map<std::string, void*> name_to_ptr_;
  std::uint64_t total_allocated_bytes_ {0};
};

} // namespace llama_qnn


