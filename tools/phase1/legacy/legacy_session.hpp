#pragma once
#include "modules/fire_control/fire_control.hpp"
#include <deque>
#include <optional>
namespace mv::modules {
using ControlStepResult = FireControlResult;
/** @brief Synchronous single-owner control chain. Step/AcknowledgePublication must alternate. */
class ControlSession final {
 public:
  ControlSession(FireControlConfig config, GimbalTrajectoryPlannerConfig planner);
  [[nodiscard]] ControlStepResult Step(const ControlInputSnapshot& input, const hal::GimbalActuatorTelemetry& actuator, bool sink_healthy, std::chrono::steady_clock::time_point now, std::uint64_t command_timestamp_ns);
  /** @brief A send acknowledgement is not a launch confirmation. Rejects duplicate/stale acknowledgements. */
  void AcknowledgePublication(ControlStepResult& result, bool succeeded, std::chrono::steady_clock::time_point now);
  void ClearPublishedProjection(std::string_view reason) noexcept;
  void Reset();
 private:
  struct LoopState {
    std::uint64_t observed_sequence = ~std::uint64_t{0};
    std::optional<bool> last_external_control;
    std::optional<bool> last_sink_healthy;
    std::optional<hal::GimbalActuatorMode> last_actuator_mode;
    std::uint64_t control_cycles{0};
    modules::MatchedGimbalCommand matched_command;
  };

  [[nodiscard]] modules::MatchedGimbalCommand MatchCommand(
      const std::optional<std::uint64_t>& capture_timestamp_ns,
      const std::optional<hal::GimbalActuatorTelemetry>& actuator) const noexcept;
  void RememberCommand(const hal::GimbalCommand& command);
  void AttachProjectionDiagnostics(modules::FireControlResult& result);

  FireControl fire_control_;
  GimbalFeedbackEstimator feedback_estimator_;
  const double PLANNER_DT_S;
  LoopState state_;
  std::optional<std::uint64_t> round_;
  std::uint64_t cycle_{0};
  bool awaiting_ack_{false};
  std::optional<std::chrono::steady_clock::time_point> last_step_time_;
  std::deque<hal::GimbalCommand> sent_commands_;
  std::vector<modules::PlannedGimbalPoint> last_successful_trajectory_;
  std::size_t last_successful_command_index_{1};
  double last_successful_target_distance_m_{-1.0};
  std::chrono::steady_clock::time_point last_successful_plan_time_{};
  int last_successful_plan_slot_{-1};
  int consecutive_mpc_failure_cycles_{0};
  bool control_projection_active_{false};
  bool output_projection_cleared_pending_{false};
  std::string output_projection_clear_reason_;
};
} // namespace mv::modules
