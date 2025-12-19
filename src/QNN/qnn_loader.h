#pragma once

#include <string>
#include <vector>
#include <QnnTensor.h>

// dlopen으로 연 QNN 공유 라이브러리 핸들 쌍을 보관
struct DlHandlePair { void* backend_so_handle; void* system_so_handle; };

namespace llama_qnn {

// Executorch 구현 흐름을 그대로 따르는 QNN 초기화/컨텍스트 복원 헬퍼
// - QnnInterface_getProviders로 provider를 받아 함수 테이블을 획득
// - backendCreate → deviceCreate 순으로 생성
// - contextCreateFromBinaryListAsync(우선) / contextCreateFromBinary(폴백)로 복원
class QnnLoader {
public:
  QnnLoader() = default;
  ~QnnLoader();

  // QNN 백엔드/시스템 so를 로드하고, QnnInterface_getProviders 심볼을 확인한다.
  // - backend_so_path: 보통 libQnnHtp.so (Android에선 /vendor/lib64/libQnnHtp.so)
  // - system_so_path:  보통 libQnnSystem.so
  bool load(const std::string& backend_so_path, const std::string& system_so_path);

  // provider 목록을 질의해 원하는 provider 하나를 선택한다(기본 첫 번째).
  const void* get_interface_provider(const char* provider_name = nullptr);
  const void* interface() const { return interface_provider_; }
  const DlHandlePair& handles() const { return handles_; }

  // 로그 레벨 설정 (1=ERROR,2=WARN,3=INFO,4=VERBOSE,5=DEBUG)
  void set_log_level(int level) { log_level_ = level; }

  // Executorch와 동일: logCreate → backendCreate → deviceCreate 순으로 핸들을 생성한다.
  bool create_backend_and_device();

  // 멀티 샤드 컨텍스트 바이너리를 한 번에 복원 (Executorch의 멀티그래프 바이너리 대응)
  // (비사용) ListAsync 경로는 제거

  // Executorch와 동일 경로: 단일 바이너리 버퍼로 컨텍스트 1개 복원
  bool create_context_from_binary(const void* binary, size_t binary_size);
  
  // Multi-context: 여러 바이너리로 컨텍스트를 순차적으로 생성
  bool create_contexts_from_binaries(const std::vector<std::pair<const void*, size_t>>& binaries);
  size_t num_contexts() const { return contexts_.size(); }

  // 컨텍스트 인덱스와 그래프 이름으로 그래프 핸들을 조회하여 내부에 보관
  bool retrieve_graph(size_t ctx_index, const std::string& graph_name);
  size_t num_graphs() const { return graphs_.size(); }

  // 그래프 실행: 입력/출력 텐서를 전달하여 graphExecute 호출
  bool execute_graph(size_t ctx_index,
                     const std::string& graph_name,
                     const std::vector<Qnn_Tensor_t>& inputs,
                     std::vector<Qnn_Tensor_t>& outputs);

  // 그래프 등록 IO 텐서 조회(Executorch 흐름: 등록된 텐서 ID 사용)
  bool get_graph_io(size_t ctx_index,
                    const std::string& graph_name,
                    std::vector<Qnn_Tensor_t>& inputs,
                    std::vector<Qnn_Tensor_t>& outputs);

  // 등록된 그래프 텐서에 clientBuf를 업데이트(Executorch AllocateTensor 유사)
  bool update_graph_tensors(size_t ctx_index,
                            const std::string& graph_name,
                            const std::vector<Qnn_Tensor_t>& tensors);

  // 생성/복원한 리소스를 역순으로 정리 (context → device → backend → logger → dlclose)
  void cleanup();

  // HTP 성능 모드 활성화 (Burst/HighPerformance)
  bool enable_htp_performance_mode();

private:
  void* get_providers_fn_ {nullptr};
  const void* interface_provider_ {nullptr};
  DlHandlePair handles_ {nullptr, nullptr};
  void* logger_ {nullptr};
  void* backend_ {nullptr};
  void* device_ {nullptr};
  std::vector<void*> contexts_;
  std::vector<void*> graphs_;

  uint32_t power_config_client_id_ {0}; // HTP Power Config Client ID

  int log_level_ {5};

  // 로깅 콜백은 SDK 버전별 시그니처 차이가 커서 생략
};

} // namespace llama_qnn


