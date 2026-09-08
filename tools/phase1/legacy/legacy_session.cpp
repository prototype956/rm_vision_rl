#include "legacy_session.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
namespace mv::modules { namespace {
struct ProjectedKinematics {
  std::optional<frame::FrameKinematics> value;
  double dt_s{0.0};
  bool chassis_motion_valid{false};
  const char* status{"not_projected"};
};

ProjectedKinematics ProjectKinematics(const modules::ControlInputSnapshot& input,
                                      const hal::GimbalFeedback& feedback,
                                      std::chrono::steady_clock::time_point now,
                                      double max_age_s) noexcept {
  const auto SOURCE_TIME = input.prediction.source_steady_time
                               ? input.prediction.source_steady_time
                               : (!input.prediction.source_capture_timestamp_ns
                                      ? std::optional(input.prediction.source_receive_steady_time)
                                      : std::nullopt);
  if (!SOURCE_TIME || *SOURCE_TIME > now) {
    ProjectedKinematics result;
    result.status = "invalid_capture_time";
    return result;
  }
  const double DT_S = std::chrono::duration<double>(now - *SOURCE_TIME).count();
  if (!std::isfinite(DT_S) || DT_S > max_age_s) {
    ProjectedKinematics result;
    result.dt_s = DT_S;
    result.status = "stale_prediction";
    return result;
  }
  if (!feedback.valid || feedback.timestamp == std::chrono::steady_clock::time_point{} ||
      feedback.timestamp > now || !std::isfinite(feedback.yaw) || !std::isfinite(feedback.pitch)) {
    ProjectedKinematics result;
    result.dt_s = DT_S;
    result.status = "invalid_feedback";
    return result;
  }
  const double FEEDBACK_AGE_S = std::chrono::duration<double>(now - feedback.timestamp).count();
  if (!std::isfinite(FEEDBACK_AGE_S) || FEEDBACK_AGE_S > max_age_s) {
    ProjectedKinematics result;
    result.dt_s = DT_S;
    result.status = "stale_feedback";
    return result;
  }

  frame::FrameKinematics projected{.world_t_gimbal = input.world_t_gimbal,
                                   .gimbal_t_camera_optical = input.gimbal_t_camera_optical,
                                   .gimbal_t_muzzle = input.gimbal_t_muzzle};
  projected.world_t_gimbal.rotation =
      Eigen::AngleAxisd(feedback.yaw, geometry::Vector3::UnitZ()) *
      Eigen::AngleAxisd(-feedback.pitch, geometry::Vector3::UnitY());

  bool chassis_valid = false;
  if (input.chassis_motion && std::isfinite(input.chassis_motion->yaw_rad) &&
      input.chassis_motion->velocity_body_mps.allFinite()) {
    const double COS_YAW = std::cos(input.chassis_motion->yaw_rad);
    const double SIN_YAW = std::sin(input.chassis_motion->yaw_rad);
    const double FORWARD = input.chassis_motion->velocity_body_mps.x();
    const double LEFT = input.chassis_motion->velocity_body_mps.y();
    projected.world_t_gimbal.translation.x() += (COS_YAW * FORWARD - SIN_YAW * LEFT) * DT_S;
    projected.world_t_gimbal.translation.y() += (SIN_YAW * FORWARD + COS_YAW * LEFT) * DT_S;
    chassis_valid = true;
  }
  if (!projected.world_t_gimbal.translation.allFinite() ||
      !projected.world_t_gimbal.rotation.coeffs().allFinite()) {
    ProjectedKinematics result;
    result.dt_s = DT_S;
    result.status = "non_finite_projection";
    return result;
  }
  return {.value = projected,
          .dt_s = DT_S,
          .chassis_motion_valid = chassis_valid,
          .status = chassis_valid ? "projected" : "position_held_no_chassis_motion"};
}

}
ControlSession::ControlSession(FireControlConfig c,GimbalTrajectoryPlannerConfig p):fire_control_(c,p),feedback_estimator_(p.max_yaw_velocity_rad_s,p.max_pitch_velocity_rad_s),PLANNER_DT_S(p.dt_s){}
modules::MatchedGimbalCommand ControlSession::MatchCommand(
    const std::optional<std::uint64_t>& capture_timestamp_ns,
    const std::optional<hal::GimbalActuatorTelemetry>& actuator) const noexcept {
  modules::MatchedGimbalCommand match;
  if (!capture_timestamp_ns)
    return match;
  if (actuator && actuator->valid && actuator->consumed_command_timestamp_ns != 0) {
    for (auto iterator = sent_commands_.rbegin(); iterator != sent_commands_.rend(); ++iterator) {
      if (iterator->timestamp_ns == actuator->consumed_command_timestamp_ns) {
        match.valid = iterator->valid;
        match.approximate = false;
        match.command = *iterator;
        match.age_at_capture_s =
            *capture_timestamp_ns >= iterator->timestamp_ns
                ? static_cast<double>(*capture_timestamp_ns - iterator->timestamp_ns) * 1.0e-9
                : 0.0;
        return match;
      }
    }
  }
  for (auto iterator = sent_commands_.rbegin(); iterator != sent_commands_.rend(); ++iterator) {
    if (iterator->timestamp_ns <= *capture_timestamp_ns) {
      match.valid = iterator->valid;
      match.approximate = true;
      match.command = *iterator;
      match.age_at_capture_s =
          static_cast<double>(*capture_timestamp_ns - iterator->timestamp_ns) * 1.0e-9;
      return match;
    }
  }
  return match;
}

void ControlSession::RememberCommand(const hal::GimbalCommand& command) {
  sent_commands_.push_back(command);
  while (!sent_commands_.empty() &&
         command.timestamp_ns > sent_commands_.front().timestamp_ns + 1'000'000'000ULL) {
    sent_commands_.pop_front();
  }
}

void ControlSession::ClearPublishedProjection(std::string_view reason) noexcept {
  feedback_estimator_.ClearCommandProjection();
  fire_control_.ResetFireReadiness();
  last_successful_trajectory_.clear();
  control_projection_active_ = false;
  output_projection_cleared_pending_ = true;
  if (!output_projection_clear_reason_.empty())
    output_projection_clear_reason_.push_back(',');
  output_projection_clear_reason_.append(reason);
}

void ControlSession::AttachProjectionDiagnostics(modules::FireControlResult& result) {
  result.diagnostics.output_projection_cleared = output_projection_cleared_pending_;
  result.diagnostics.output_projection_clear_reason = std::move(output_projection_clear_reason_);
  output_projection_cleared_pending_ = false;
  output_projection_clear_reason_.clear();
}

ControlStepResult ControlSession::Step(const ControlInputSnapshot& sample, const hal::GimbalActuatorTelemetry& ACTUATOR, bool SINK_HEALTHY, std::chrono::steady_clock::time_point now, std::uint64_t SYSTEM_NOW_NS) {
const auto* snapshot=&sample; auto& state=state_;
  const std::optional<hal::GimbalActuatorMode> ACTUATOR_MODE =
      ACTUATOR.valid ? std::optional(ACTUATOR.mode) : std::nullopt;
  if (ACTUATOR_MODE != state.last_actuator_mode) {
    feedback_estimator_.ClearRuntimeActuator();
    ClearPublishedProjection("actuator_mode_changed");
    state.last_actuator_mode = ACTUATOR_MODE;
  }

  auto input = *snapshot;
  input.external_control_enabled = snapshot->external_control_enabled;
  if (!state.last_external_control ||
      *state.last_external_control != input.external_control_enabled) {
    if (input.external_control_enabled) {
      MV_LOG_INFO("Control", "Talos external auto-aim subscription enabled");
    } else {
      MV_LOG_WARN("Control",
                  "Talos external auto-aim subscription is disabled; press F5 in simulation");
    }
    feedback_estimator_.ClearRuntimeActuator();
    ClearPublishedProjection(input.external_control_enabled ? "external_control_enabled"
                                                            : "external_control_disabled");
    state.last_external_control = input.external_control_enabled;
  }

  const bool RUNTIME_WAS_ACTIVE = feedback_estimator_.RuntimeActuatorActive();
  feedback_estimator_.ObserveActuatorTelemetry(ACTUATOR, now, SYSTEM_NOW_NS);
  if (RUNTIME_WAS_ACTIVE != feedback_estimator_.RuntimeActuatorActive()) {
    ClearPublishedProjection(feedback_estimator_.RuntimeActuatorActive()
                                 ? "runtime_actuator_enabled"
                                 : "runtime_actuator_invalid");
  }

  bool measurement_fresh = false;
  if (snapshot->prediction.sequence != state.observed_sequence) {
    if (snapshot->prediction.source_steady_time) {
      feedback_estimator_.ObserveMeasurement(snapshot->prediction.sequence,
                                             *snapshot->prediction.source_steady_time,
                                             snapshot->world_t_gimbal, snapshot->frame_actuator);
      measurement_fresh = true;
    }
    state.observed_sequence = snapshot->prediction.sequence;
    state.matched_command =
        MatchCommand(snapshot->prediction.source_capture_timestamp_ns, snapshot->frame_actuator);
  }
  if (!state.last_sink_healthy || *state.last_sink_healthy != SINK_HEALTHY) {
    if (!SINK_HEALTHY)
      ClearPublishedProjection("talos_unhealthy");
    state.last_sink_healthy = SINK_HEALTHY;
  }
  const auto FEEDBACK = feedback_estimator_.Estimate(now);
  const auto FEEDBACK_SOURCE = feedback_estimator_.Source();
  const auto PROJECTED =
      ProjectKinematics(input, FEEDBACK, now, fire_control_.Config().max_prediction_age_s);
  if (PROJECTED.value)
    input.world_t_gimbal = PROJECTED.value->world_t_gimbal;
  auto result = fire_control_.Step(input, FEEDBACK, now);
  auto& output = result.output;
  auto& diagnostics = result.diagnostics;
  if (output.tracking_object_reset && control_projection_active_) {
    feedback_estimator_.ClearRuntimeActuator();
    ClearPublishedProjection("tracking_object_changed");
  }
  diagnostics.feedback_source = FEEDBACK_SOURCE;
  diagnostics.measured_feedback = feedback_estimator_.LastMeasurement();
  diagnostics.measurement_fresh = measurement_fresh;
  diagnostics.measurement_age_s =
      diagnostics.measured_feedback.valid
          ? std::max(0.0,
                     std::chrono::duration<double>(now - diagnostics.measured_feedback.timestamp)
                         .count())
          : std::numeric_limits<double>::infinity();
  diagnostics.matched_prior_command = state.matched_command;
  diagnostics.actuator_telemetry = ACTUATOR;
  diagnostics.frame_actuator_telemetry = snapshot->frame_actuator;
  diagnostics.runtime_actuator_age_s = feedback_estimator_.RuntimeActuatorAgeS();
  diagnostics.feedback_projection_dt_s = feedback_estimator_.ProjectionDtS();
  diagnostics.pose_projection_dt_s = PROJECTED.dt_s;
  diagnostics.chassis_motion_valid = PROJECTED.chassis_motion_valid;
  diagnostics.pose_projection_status = PROJECTED.status;
  diagnostics.projected_kinematics = PROJECTED.value;
  diagnostics.feedback_runtime_state_timestamp_ns = feedback_estimator_.RuntimeStateTimestampNs();
  if (snapshot->frame_actuator && snapshot->frame_actuator->valid &&
      snapshot->frame_actuator->state_timestamp_ns != 0 &&
      snapshot->frame_actuator->state_timestamp_ns <= SYSTEM_NOW_NS) {
    diagnostics.frame_actuator_age_s =
        static_cast<double>(SYSTEM_NOW_NS - snapshot->frame_actuator->state_timestamp_ns) * 1.0e-9;
  } else {
    diagnostics.frame_actuator_age_s = std::numeric_limits<double>::infinity();
  }
  diagnostics.feedback_runtime_comparison_valid =
      FEEDBACK.valid && ACTUATOR.valid && ACTUATOR.mode == hal::GimbalActuatorMode::PHYSICAL;
  if (diagnostics.feedback_runtime_comparison_valid) {
    diagnostics.yaw_feedback_minus_runtime_actuator =
        std::remainder(FEEDBACK.yaw - ACTUATOR.actual_yaw, 2.0 * std::numbers::pi);
    diagnostics.pitch_feedback_minus_runtime_actuator = FEEDBACK.pitch - ACTUATOR.actual_pitch;
  }
  diagnostics.frame_runtime_comparison_valid =
      snapshot->frame_actuator && snapshot->frame_actuator->valid && ACTUATOR.valid &&
      snapshot->frame_actuator->mode == hal::GimbalActuatorMode::PHYSICAL &&
      ACTUATOR.mode == hal::GimbalActuatorMode::PHYSICAL;
  if (diagnostics.frame_runtime_comparison_valid) {
    const auto& frame = *snapshot->frame_actuator;
    diagnostics.yaw_frame_minus_runtime_actuator =
        std::remainder(frame.actual_yaw - ACTUATOR.actual_yaw, 2.0 * std::numbers::pi);
    diagnostics.pitch_frame_minus_runtime_actuator = frame.actual_pitch - ACTUATOR.actual_pitch;
    diagnostics.yaw_frame_acceleration_minus_runtime =
        frame.yaw_acceleration - ACTUATOR.yaw_acceleration;
    diagnostics.pitch_frame_acceleration_minus_runtime =
        frame.pitch_acceleration - ACTUATOR.pitch_acceleration;
  }
  output.command_sink_healthy = SINK_HEALTHY;


  if (output.reject_reason == modules::FireRejectReason::MPC_FAILED) {
    ++consecutive_mpc_failure_cycles_;
  } else {
    consecutive_mpc_failure_cycles_ = 0;
  }
  output.consecutive_mpc_failure_cycles = consecutive_mpc_failure_cycles_;

  if (output.reject_reason == modules::FireRejectReason::MPC_FAILED &&
      !last_successful_trajectory_.empty() && output.external_control_enabled &&
      output.command_sink_healthy && !output.tracking_object_reset &&
      output.selected_slot == last_successful_plan_slot_) {
    const double FALLBACK_AGE_S =
        std::max(0.0, std::chrono::duration<double>(now - last_successful_plan_time_).count());
    output.fallback_age_s = FALLBACK_AGE_S;
    output.fallback_source_slot = last_successful_plan_slot_;
    constexpr double MAX_FALLBACK_AGE_S = 0.100;
    const auto ELAPSED_STEPS =
        static_cast<std::size_t>(std::max(1LL, std::llround(FALLBACK_AGE_S / PLANNER_DT_S)));
    const auto FALLBACK_INDEX = last_successful_command_index_ + ELAPSED_STEPS;
    if (FALLBACK_AGE_S <= MAX_FALLBACK_AGE_S &&
        FALLBACK_INDEX < last_successful_trajectory_.size()) {
      const auto& point = last_successful_trajectory_[FALLBACK_INDEX];
      output.command = {.valid = true,
                        .fire = false,
                        .timestamp_ns = output.command_timestamp_ns,
                        .yaw = std::remainder(point.yaw, 2.0 * std::numbers::pi),
                        .yaw_velocity = point.yaw_velocity,
                        .yaw_acceleration = point.yaw_acceleration,
                        .pitch = point.pitch,
                        .pitch_velocity = point.pitch_velocity,
                        .pitch_acceleration = point.pitch_acceleration,
                        .target_distance_m = last_successful_target_distance_m_};
      output.command.valid = true;
      output.command.fire = false;
      output.command_source = modules::GimbalCommandSource::TRAJECTORY_FALLBACK;
      output.fallback_active = true;
      output.fallback_trajectory_index = static_cast<int>(FALLBACK_INDEX);
      output.fallback_remaining_points =
          static_cast<int>(last_successful_trajectory_.size() - FALLBACK_INDEX - 1);
    }
  }

  if (!output.command_sink_healthy || !output.external_control_enabled) {
    output.command.valid = false;
    output.command.fire = false;
    output.command_source = modules::GimbalCommandSource::STOP;
    output.reject_reason = modules::FireRejectReason::TALOS_UNHEALTHY;
    if (!output.external_control_enabled)
      output.reject_reason = modules::FireRejectReason::EXTERNAL_CONTROL_DISABLED;
  }

  if (!output.command.valid) {
    output.command_source = modules::GimbalCommandSource::STOP;
    output.command.fire = false;
    if (control_projection_active_) {
      if (output.reject_reason == modules::FireRejectReason::MPC_FAILED) {
        diagnostics.fallback_expired_this_cycle = true;
        ClearPublishedProjection("mpc_fallback_expired");
      } else {
        ClearPublishedProjection(modules::FireRejectReasonName(output.reject_reason));
      }
    }
  }
  output.command.source_round_id=sample.prediction.source_round_id;
  output.command.source_frame_sequence=sample.prediction.sequence;
  output.command.source_capture_timestamp_ns=sample.prediction.source_capture_timestamp_ns.value_or(0);
  return result;
}
void ControlSession::AcknowledgePublication(ControlStepResult& result,bool SEND_SUCCEEDED,std::chrono::steady_clock::time_point now){auto& output=result.output;
  output.command_publish_succeeded = SEND_SUCCEEDED && output.command.valid;
  output.published_valid = output.command_publish_succeeded;
  if (!SEND_SUCCEEDED) {
    output.command_sink_healthy = false;
    output.command.valid = false;
    output.command.fire = false;
    output.reject_reason = modules::FireRejectReason::TALOS_UNHEALTHY;
    output.command_source = modules::GimbalCommandSource::STOP;
    output.published_valid = false;
    ClearPublishedProjection("command_send_failed");
  }
  if (SEND_SUCCEEDED)
    RememberCommand(output.command);
  if (output.command_publish_succeeded) {
    feedback_estimator_.ObservePublishedCommand(output.command, now, false);
    control_projection_active_ = true;
    if (output.command_source == modules::GimbalCommandSource::MPC) {
      last_successful_trajectory_ = output.plan.trajectory;
      last_successful_command_index_ = static_cast<std::size_t>(output.plan.command_index);
      last_successful_target_distance_m_ = output.command.target_distance_m;
      last_successful_plan_time_ = now;
      last_successful_plan_slot_ = output.selected_slot;
    }
  }
  AttachProjectionDiagnostics(result);
}
}
