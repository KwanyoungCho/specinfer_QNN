#include "binary_provider.h"

#include <QnnContext.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace llama_qnn {

namespace {
static bool mmap_file(const std::string& path, std::unique_ptr<MappingOwner>& owner) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  struct stat st{};
  if (fstat(fd, &st) != 0) { close(fd); return false; }
  void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) { close(fd); return false; }
  owner.reset(new MappingOwner());
  owner->addr = addr;
  owner->size = static_cast<size_t>(st.st_size);
  owner->fd = fd;
  return true;
}
}
MappingOwner::~MappingOwner() {
  if (addr && size) munmap(addr, size);
  if (fd >= 0) close(fd);
}


// ctx_out 디렉터리 경로 보관
FileShardProvider::FileShardProvider(const std::string& ctx_dir) : dir_(ctx_dir) {}

// 선호 프리픽스 우선순위대로 연속된 index의 .bin 파일을 스캔하여 가장 많은 세트를 선택
bool FileShardProvider::init_from_dir(const std::vector<std::string>& preferred_prefixes) {
  std::vector<Shard> best;
  size_t max_count = 0;
  for (const auto& prefix : preferred_prefixes) {
    std::vector<Shard> current;
    for (size_t i = 0;; ++i) {
      fs::path p = fs::path(dir_) / (prefix + std::to_string(i) + ".bin");
      if (!fs::exists(p)) break;
      current.push_back(Shard{p.string(), fs::file_size(p)});
    }
    if (current.size() > max_count) {
      max_count = current.size();
      best.swap(current);
    }
  }
  if (best.empty()) {
    std::cerr << "No shards found under " << dir_ << "\n";
    return false;
  }
  shards_.swap(best);
  return true;
}

// 각 shard를 메모리로 읽고 QnnContext_Params_t(v1) 구조로 감싸
// contextCreateFromBinaryListAsync/FromBinary 호출에 전달할 포인터 배열을 만든다
bool FileShardProvider::build_params(std::vector<void*>& out_params,
                                     std::vector<std::unique_ptr<MappingOwner>>& out_owners) {
  out_params.clear();
  out_owners.clear();
  out_params.reserve(shards_.size() + 1);
  out_owners.reserve(shards_.size());

  for (const auto& s : shards_) {
    std::unique_ptr<MappingOwner> map_owner;
    if (!mmap_file(s.path, map_owner)) {
      std::cerr << "Failed to mmap shard: " << s.path << "\n";
      return false;
    }

    // QnnContext_Params_t는 내부에 const 멤버가 있어 배치 new로 v1을 구성
    auto* params = static_cast<QnnContext_Params_t*>(std::malloc(sizeof(QnnContext_Params_t)));
    if (!params) { std::cerr << "OOM allocating QnnContext_Params_t\n"; return false; }
    new (&params->v1) QnnContext_ParamsV1_t{ /*config*/nullptr, /*binaryBuffer*/map_owner->addr,
      static_cast<Qnn_ContextBinarySize_t>(map_owner->size), /*profile*/nullptr,
      /*notifyFunc*/nullptr, /*notifyParam*/nullptr };
    params->version = QNN_CONTEXT_PARAMS_VERSION_1;

    out_params.push_back(reinterpret_cast<void*>(params));
    out_owners.push_back(std::move(map_owner));
  }
  out_params.push_back(nullptr); // NULL-terminate
  return true;
}

} // namespace llama_qnn


