#include "io_alloc.h"

#include <cstdlib>

namespace llama_qnn {

// [설명]
// - ExecuTorch의 QnnManager::AllocateTensor 흐름을 참고하되,
//   현재 단계에서는 mutable buffer 공유가 없으므로 텐서별(per‑tensor) 할당만 수행한다.
// - 추후 MethodMeta 기반 플랜을 적용하면 동일 인터페이스에서 공유 버퍼로 확장 가능.

void QNNIOAllocator::build_from_qnnjson(const QnnJsonGraphDesc& desc) {
  name_to_nbytes_.clear();
  name_to_ptr_.clear();
  total_allocated_bytes_ = 0;

  // 입력 텐서
  for (const auto& t : desc.inputs) {
    name_to_nbytes_[t.name] = t.nbytes;
  }
  // 출력 텐서
  for (const auto& t : desc.outputs) {
    name_to_nbytes_[t.name] = t.nbytes;
  }
}

std::uint64_t QNNIOAllocator::allocate(std::size_t alignment) {
  // 기존 버퍼 해제 후 새로 할당
  release();
  total_allocated_bytes_ = 0;

  for (const auto& kv : name_to_nbytes_) {
    const std::string& name = kv.first;
    std::size_t sz = static_cast<std::size_t>(kv.second);
    if (sz == 0) {
      name_to_ptr_[name] = nullptr;
      continue;
    }
    void* p = nullptr;
    if (is_pow2(alignment)) {
      if (posix_memalign(&p, alignment, sz) != 0) p = nullptr;
    } else {
      p = std::malloc(sz);
    }
    name_to_ptr_[name] = p;
    if (p) total_allocated_bytes_ += kv.second;
  }
  return total_allocated_bytes_;
}

void QNNIOAllocator::release() {
  for (auto& kv : name_to_ptr_) {
    if (kv.second) std::free(kv.second);
    kv.second = nullptr;
  }
  name_to_ptr_.clear();
  total_allocated_bytes_ = 0;
}

} // namespace llama_qnn


