#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llama_qnn {

// mmapped 파일을 생명주기 끝까지 유지하기 위한 소유자
class MappingOwner {
public:
  void* addr {nullptr};
  size_t size {0};
  int fd {-1};
  ~MappingOwner();
};

// 컨텍스트 바이너리(.bin) 샤드 파일을 디렉터리에서 검색/적재하여
// QnnContext_Params_t 리스트로 만들어 주는 유틸리티
class FileShardProvider {
public:
  struct Shard {
    std::string path;
    uint64_t size_bytes {0};
  };

  // ctx_out 디렉터리를 기반으로, 선호 프리픽스 우선순위대로 연속된 index의 .bin 파일을 수집
  explicit FileShardProvider(const std::string& ctx_dir);
  // 예: {"kv_forward_prefill_forward_", "kv_forward_"}
  bool init_from_dir(const std::vector<std::string>& preferred_prefixes);
  // 각 shard를 mmmap하여 QnnContext_Params_t(v1)를 구성하고, 마지막에 NULL-terminate
  // owners: mmapped 영역을 보관(스코프 종료 시 자동 unmap/close)
  bool build_params(std::vector<void*>& out_params,
                    std::vector<std::unique_ptr<MappingOwner>>& out_owners);
  const std::vector<Shard>& shards() const { return shards_; }

private:
  std::string dir_;
  std::vector<Shard> shards_;
};

} // namespace llama_qnn


