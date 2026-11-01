#pragma once
#include <nlohmann/json.hpp>

#include "tool/foxglove/offline_recorder.hpp"
#include "tool/simulation_evaluation/simulation_evaluator.hpp"
namespace rmvision_rl {
/** @brief Independent post-computation diagnostics; never exposed to policy callbacks. */
class ReplayDiagnostics final {
 public:
  ReplayDiagnostics(const std::string& path, nlohmann::json info,
                    mv::modules::ArmorPnpConfig config);
  void Frame(mv::tool::foxglove::pipeline::VisionDebugFrame frame);
  void Control(const mv::modules::FireControlResult& result);
  void Snapshot(const nlohmann::json& payload);
  void Close(const nlohmann::json& summary);

 private:
  mv::tool::foxglove::OfflineRecorder recorder_;
  mv::tool::simulation_evaluation::SimulationEvaluator evaluator_;
  nlohmann::json info_;
  std::vector<mv::tool::foxglove::pipeline::VisionDebugFrame> frames_;
  std::optional<mv::modules::FireControlResult> control_;
  std::uint64_t last_ns_{0};
  std::uint64_t control_samples_{0}, vision_samples_{0};
};
}  // namespace rmvision_rl
