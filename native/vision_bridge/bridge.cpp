#include "core/logger.hpp"
#include "modules/armor_pnp/armor_pnp.hpp"
#include "modules/armor_predictor/armor_predictor.hpp"
#include "modules/fire_control/control_session.hpp"
#include "modules/fire_control/fire_only_policy_adapter.hpp"
#include "policy_wire.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include <nlohmann/json.hpp>
#include <numbers>

using Json = nlohmann::json;
namespace {
using namespace mv;
constexpr std::uint64_t EPOCH = 1'000'000'000'000'000'000ULL;
// 退出未提交的策略周期，不继续计算 MPC 或发布命令。
struct PolicyCancelled {};
auto Time(std::uint64_t ns) {
  return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}
void Require(bool valid, const char* message) {
  if (!valid)
    throw std::runtime_error(message);
}
void Keys(const Json& value, std::initializer_list<std::string_view> allowed) {
  Require(value.is_object(), "object required");
  for (const auto& [key, unused] : value.items()) {
    (void)unused;
    Require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(),
            "unexpected input field");
  }
}
geometry::RigidTransform Transform(const Json& value) {
  geometry::RigidTransform result;
  const auto t = value.at("translation").get<std::array<double, 3>>();
  const auto q = value.at("rotation_xyzw").get<std::array<double, 4>>();
  result.translation = {t[0], t[1], t[2]};
  result.rotation = Eigen::Quaterniond(q[3], q[0], q[1], q[2]);
  Require(result.translation.allFinite() && result.rotation.coeffs().allFinite() &&
              std::abs(result.rotation.norm() - 1.0) < 1e-4,
          "invalid platform transform");
  result.rotation.normalize();
  return result;
}
hal::GimbalActuatorTelemetry Feedback(const Json& value) {
  hal::GimbalActuatorTelemetry result;
  result.valid = value.at("valid");
  const int mode = value.at("mode");
  Require(mode == 1 || mode == 2, "invalid actuator mode");
  result.mode = static_cast<hal::GimbalActuatorMode>(mode);
  result.state_timestamp_ns = value.at("timestamp_ns");
  result.consumed_command_timestamp_ns = value.at("consumed_command_timestamp_ns");
  result.consumed_at_timestamp_ns = value.at("consumed_at_timestamp_ns");
  result.target_yaw = value.at("target_yaw_rad");
  result.target_pitch = value.at("target_pitch_rad");
  result.yaw_acceleration = value.at("yaw_acceleration_rad_s2");
  result.pitch_acceleration = value.at("pitch_acceleration_rad_s2");
  result.command_valid = value.at("command_valid");
  result.saturation_flags = value.at("saturation_flags");
  result.actual_yaw = value.at("yaw_rad");
  result.actual_pitch = value.at("pitch_rad");
  result.yaw_velocity = value.at("yaw_velocity_rad_s");
  result.pitch_velocity = value.at("pitch_velocity_rad_s");
  return result;
}

/**
 * @brief 同步管理估计器与控制器，输入仅包含检测帧和自身反馈等控制数据。
 *
 * 每次控制计算生成待确认结果，由调用方提交给仿真后通过 ack 确认发布。
 * 策略回调可在同一次计算内等待 Python 动作，等待期间不推进仿真。
 */
class Bridge {
 public:
  explicit Bridge(const std::string& root, std::ostream* diagnostics = nullptr)
      : root_(root), diagnostics_(diagnostics) {}
  /** @brief 丢弃未提交周期并使控制器失效，后续请求必须先 Reset。 */
  void Discard() {
    pending_.reset();
    control_.reset();  // 部分更新的周期不可复用，下次必须 Reset。
    training_ = false;
  }
  /**
   * @brief 处理一个 JSONL 请求，维护回合、控制周期和命令确认状态。
   * @param request 已解析的请求对象，op 指定操作类型。
   * @return 操作结果；外部策略模式可能先通过标准输出发送观测并等待动作。
   */
  Json Handle(const Json& request) {
    const std::string op = request.at("op");
    if (op == "reset") {
      Keys(request, {"op", "round_id"});
      round_ = request.at("round_id");
      pnp_ = std::make_unique<modules::ArmorPnp>(modules::ParseArmorPnpConfig(Load("armor_pnp")));
      predictor_ = std::make_unique<modules::ArmorPredictor>(
          modules::ParseArmorPredictorConfig(Load("armor_predictor")));
      control_ = std::make_unique<modules::ControlSession>(
          modules::ParseFireControlConfig(Load("fire_control")),
          modules::ParseGimbalTrajectoryPlannerConfig(Load("gimbal_trajectory_planner")));
      input_ = {};
      input_.prediction.source_round_id = round_;
      input_.prediction.sequence = std::numeric_limits<std::uint64_t>::max();
      pending_.reset();
      last_tick_.reset();
      last_frame_.reset();
      search_start_.reset();
      evaluation_window_.reset();
      training_ = false;
      policy_mode_ = "rule";
      fire_only_ = std::make_unique<modules::FireOnlyPolicyAdapter>(
          modules::ParseFireControlConfig(Load("fire_control")));
      processed_ = 0;
      return {{"ok", true}};
    }
    Require(control_ != nullptr, "reset required");
    if (op == "cancel") {
      Keys(request, {"op"});
      Require(pending_.has_value(), "no prepared command to cancel");
      Discard();
      return {{"ok", true}, {"kind", "cancelled"}};
    }
    if (op == "begin_training") {
      Keys(request, {"op", "round_id", "start_ns"});
      Require(!pending_ && last_tick_ && !evaluation_window_ && !training_,
              "training requires an acknowledged warmup");
      Require(request.at("round_id").get<std::uint64_t>() == round_ &&
                  request.at("start_ns").get<std::uint64_t>() == *last_tick_ + 10'000'000,
              "invalid training start");
      training_ = true;
      policy_mode_ = "fire_only";
      return {{"ok", true}};
    }
    if (op == "begin_evaluation") {
      Keys(request, {"op", "round_id", "start_ns", "end_ns", "policy_mode"});
      const std::string mode = request.value("policy_mode", "rule");
      Require(mode == "rule" || mode == "nine" || mode == "fire_only", "invalid policy mode");
      Require(!pending_ && last_tick_ && !evaluation_window_ && !training_,
              "evaluation requires an acknowledged warmup");
      Require(request.at("round_id").get<std::uint64_t>() == round_, "old round evaluation");
      const auto start = request.at("start_ns").get<std::uint64_t>();
      const auto end = request.at("end_ns").get<std::uint64_t>();
      Require(start == *last_tick_ + 10'000'000 && end > start &&
                  end - start <= 120'000'000'000ULL && (end - start) % 10'000'000 == 0,
              "invalid evaluation window");
      evaluation_window_ = std::pair{start, end};
      policy_mode_ = mode;
      return {{"ok", true}};
    }
    if (op == "ack") {
      Keys(request, {"op", "success"});
      Require(pending_.has_value(), "no pending publication");
      control_->AcknowledgePublication(*pending_, request.at("success"), Time(*last_tick_));
      pending_.reset();
      return {{"ok", true}};
    }
    Require(op == "step" || op == "step_policy", "unknown bridge operation");
    const bool external = op == "step_policy";
    Require(external == ((training_ || evaluation_window_.has_value()) && policy_mode_ != "rule"),
            "step operation does not match evaluation policy mode");
    Keys(request, {"op", "round_id", "sim_time_ns", "visual_frames", "feedback", "self_referee"});
    Require(!pending_, "ack previous publication first");
    Require(request.at("round_id").get<std::uint64_t>() == round_, "old round input");
    const auto now = request.at("sim_time_ns").get<std::uint64_t>();
    Require(!evaluation_window_ || now < evaluation_window_->second,
            "evaluation deadline reached; settle or reset");
    Require(!last_tick_ || now == *last_tick_ + 10'000'000,
            "control requires consecutive 10 ms ticks");
    const auto step_start = std::chrono::steady_clock::now();
    Json frame_results = Json::array();
    for (const auto& f : request.at("visual_frames")) {
      Keys(f, {"round_id", "sequence", "capture_time_ns", "capture_timestamp_ns",
               "delivery_time_ns", "camera", "kinematics", "feedback", "detections"});
      const auto sequence = f.at("sequence").get<std::uint64_t>();
      const auto capture = f.at("capture_time_ns").get<std::uint64_t>();
      Require(f.at("round_id").get<std::uint64_t>() == round_ && capture <= now &&
                  f.at("delivery_time_ns").get<std::uint64_t>() <= now &&
                  f.at("delivery_time_ns").get<std::uint64_t>() >= capture &&
                  f.at("capture_timestamp_ns").get<std::uint64_t>() == EPOCH + capture,
              "old/future frame");
      Require(!last_frame_ || sequence > *last_frame_, "duplicate/out-of-order frame");
      const auto& k = f.at("camera");
      frame::CameraModel camera;
      camera.width = k.at("width");
      camera.height = k.at("height");
      camera.fx = k.at("fx");
      camera.fy = k.at("fy");
      camera.cx = k.at("cx");
      camera.cy = k.at("cy");
      camera.distortion = k.at("distortion").get<std::array<double, 5>>();
      const auto& motion = f.at("kinematics");
      input_.world_t_gimbal = Transform(motion.at("world_t_gimbal"));
      input_.gimbal_t_camera_optical = Transform(motion.at("gimbal_t_camera_optical"));
      input_.gimbal_t_muzzle = Transform(motion.at("gimbal_t_muzzle"));
      input_.frame_actuator = Feedback(f.at("feedback"));
      std::vector<modules::ArmorDetection> detections;
      std::vector<modules::CornerRefinementOutput> refinements;
      for (const auto& d : f.at("detections")) {
        Keys(d, {"label", "color", "objectness", "corners"});
        modules::ArmorDetection detection;
        const int label = d.at("label");
        const int color = d.at("color");
        Require(label >= 0 && label <= 8 && color >= 0 && color <= 1, "invalid detection class");
        detection.label = static_cast<modules::ArmorLabel>(label);
        detection.color = static_cast<modules::ArmorColor>(color);
        detection.objectness = d.at("objectness");
        const auto corners = d.at("corners").get<std::array<std::array<float, 2>, 4>>();
        for (std::size_t i = 0; i < 4; ++i) {
          Require(std::isfinite(corners[i][0]) && std::isfinite(corners[i][1]), "nonfinite corner");
          detection.corners[i] = {corners[i][0], corners[i][1]};
        }
        detections.push_back(detection);
        refinements.push_back({.corners = detection.corners, .refined = false});
      }
      frame::FrameStamp stamp;
      stamp.simulation_round_id = round_;
      stamp.sequence = sequence;
      stamp.receive_steady_time = Time(now);
      stamp.capture_steady_time = Time(capture);
      stamp.capture_timestamp_ns = f.at("capture_timestamp_ns");
      const frame::SpatialFrameView spatial{camera, input_.world_t_gimbal,
                                            input_.gimbal_t_camera_optical, input_.gimbal_t_muzzle};
      const auto pnp = pnp_->ProcessFrame(sequence, camera, detections, refinements);
      auto prediction =
          predictor_->ProcessFrame(stamp, spatial, detections, refinements, pnp.output, {});
      input_.prediction = std::move(prediction.output);
      Json poses = Json::array();
      const auto world_camera =
          geometry::Compose(input_.world_t_gimbal, input_.gimbal_t_camera_optical);
      for (const auto& p : pnp.output.estimates) {
        const auto center = geometry::TransformPoint(world_camera, p.camera_t_armor.translation);
        poses.push_back({{"input_index", p.input_index},
                         {"center_world", {center.x(), center.y(), center.z()}}});
      }
      frame_results.push_back(
          {{"sequence", sequence},
           {"capture_time_ns", capture},
           {"detection_count", detections.size()},
           {"pnp", poses},
           {"tracker_state", modules::TrackerStateName(input_.prediction.state)}});
      last_frame_ = sequence;
      ++processed_;
    }
    const auto& r = request.at("self_referee");
    input_.referee.valid = r.at("valid");
    input_.referee.alive = r.at("hp").get<int>() > 0;
    input_.referee.fire_permitted = r.at("fire_permitted");
    input_.referee.unlimited = r.at("allowance_mode") == 0;
    input_.referee.allowance_remaining = r.at("allowance_remaining");
    input_.referee.heat = r.at("heat");
    input_.referee.heat_limit = r.at("heat_limit");
    input_.referee.cooling_per_second = r.at("cooling_per_second");
    input_.referee.fire_blocks = r.at("fire_blocks");
    input_.referee.sample_sequence = r.at("sequence");
    input_.referee.sample_time_ns = r.at("sample_ns");
    input_.referee.received_at = Time(now);
    Require(input_.referee.sample_time_ns <= now, "future referee sample");
    input_.referee.age_at_receive_s =
        static_cast<double>(now - input_.referee.sample_time_ns) * 1e-9;
    input_.external_control_enabled = true;
    const auto feedback = Feedback(request.at("feedback"));
    Json policy_record = nullptr;
    if (external) {
      pending_ = control_->StepWithPolicy(
          input_, feedback, true, Time(now), EPOCH + now,
          [&](const modules::ControlInputSnapshot& snapshot,
              const modules::PolicyObservation& value) {
            auto observation = value;
            if (policy_mode_ == "fire_only") {
              const auto choice = fire_only_->Decide(snapshot, observation, Time(now), false);
              modules::FireOnlyPolicyAdapter::RestrictMask(observation, choice);
            }
            const auto token = ++policy_token_;
            policy_record = {{"observation", rmvision_rl::Observation(observation)},
                             {"metadata",
                              {{"track_generation", snapshot.prediction.track_generation},
                               {"source_sequence", snapshot.prediction.sequence}}}};
            // 在当前回调内暂停，恢复时不再次调用 Step，避免重复融合或求解 MPC。
            std::cout << Json({{"ok", true},
                               {"kind", "policy_observation"},
                               {"token", token},
                               {"observation", policy_record.at("observation")},
                               {"metadata", policy_record.at("metadata")}})
                             .dump()
                      << std::endl;
            std::string line;
            Require(static_cast<bool>(std::getline(std::cin, line)) && line.size() <= 4096,
                    "policy response missing or too large");
            const auto reply = Json::parse(line);
            Keys(reply, {"op", "token", "action"});
            Require(reply.at("token").is_number_unsigned() &&
                        reply.at("token").get<std::uint64_t>() == token,
                    "stale policy response");
            if (reply.at("op") == "policy_cancel") {
              Keys(reply, {"op", "token"});
              throw PolicyCancelled{};
            }
            Require(reply.at("op") == "policy_action", "invalid policy response");
            const auto& action = reply.at("action");
            Require(action.is_number_integer() && action >= 0 && action <= 8,
                    "invalid policy action");
            const int index = action.get<int>();
            Require(observation.action_mask[index], "masked policy action");
            policy_record["action"] = index;
            return modules::PolicyDecision{index};
          });
      if (policy_record.is_null()) {
        // 前置条件不满足时跳过回调，清空旧的规则选板状态。
        fire_only_ = std::make_unique<modules::FireOnlyPolicyAdapter>(
            modules::ParseFireControlConfig(Load("fire_control")));
      }
    } else {
      pending_ = control_->Step(input_, feedback, true, Time(now), EPOCH + now);
    }
    auto& output = pending_->output;
    const bool raw_fire = output.command.fire;
    // 目标完全 LOST 时没有策略回调，此时才允许外部策略模式使用自身反馈执行搜索。
    // 外部 WAIT 或选定槽位求解失败时，不能用搜索命令覆盖策略结果。
    const bool lost_search = external && input_.prediction.state == modules::TrackerState::LOST &&
                             feedback.valid && (!input_.referee.valid || input_.referee.alive);
    const bool search = (!external || lost_search) && !output.command.valid;
    // 搜索仅使用自身反馈。预热阶段禁射；训练与限时评估发布共享控制器的结果，
    // 再通过发布确认更新控制状态。
    if (search) {
      if (!search_start_) {
        search_start_ = now;
        search_origin_yaw_ = feedback.actual_yaw;
      }
      output.command.valid = true;
      // 按绝对仿真时间生成扫描角度，避免执行器滞后使扫描速度逐步减小。
      // 每次搜索以实测自身偏航角为起点，实际运动仍受机械约束限制。
      constexpr double SEARCH_YAW_RATE_RAD_S = 0.6;
      const double elapsed = static_cast<double>(now - *search_start_) * 1e-9;
      output.command.yaw = std::remainder(search_origin_yaw_ + SEARCH_YAW_RATE_RAD_S * elapsed,
                                          2 * std::numbers::pi);
      output.command.pitch = 0.0;
      output.command.target_distance_m = 4.0;
    } else {
      search_start_.reset();
    }
    const bool firing_enabled = training_ || (evaluation_window_ && now >= evaluation_window_->first);
    output.command.fire = firing_enabled && !search &&
                          input_.prediction.state == modules::TrackerState::TRACKING && raw_fire;
    output.command.timestamp_ns = EPOCH + now;
    last_tick_ = now;
    const auto& center = input_.prediction.center_world;
    Json response = {
        {"ok", true},
        {"processed_frames", processed_},
        {"frames", frame_results},
        {"tracker_state", modules::TrackerStateName(input_.prediction.state)},
        {"prediction_sequence", last_frame_ ? Json(*last_frame_) : Json(nullptr)},
        {"prediction_center_world", {center.x(), center.y(), center.z()}},
        {"control",
         {{"search", search},
          {"raw_fire", raw_fire},
          {"selected_slot", output.selected_slot},
          {"reject_reason", static_cast<int>(output.reject_reason)}}},
        {"command",
         {{"valid", output.command.valid},
          {"fire", output.command.fire},
          {"yaw_rad", output.command.yaw},
          {"pitch_rad", output.command.pitch},
          {"distance_m", output.command.valid ? output.command.target_distance_m : 0.0}}}};
    if (external) {
      response["policy"] = policy_record;
      response["control"]["shot_requested"] = output.shot_requested;
      response["control"]["shot_accepted"] = output.shot_accepted;
      response["control"]["reject_reason_name"] =
          modules::FireRejectReasonName(output.reject_reason);
    }
    if (diagnostics_) {
      const auto& d = pending_->diagnostics;
      const auto& p = d.plan;
      auto axis = [](const modules::MpcAxisDiagnostics& a) {
        return Json{{"status", a.status},
                    {"solved", a.solved},
                    {"iterations", a.iterations},
                    {"primal_state", a.primal_residual_state},
                    {"primal_input", a.primal_residual_input},
                    {"dual_state", a.dual_residual_state},
                    {"dual_input", a.dual_residual_input}};
      };
      // 宿主机耗时和求解器状态仅写入可选诊断文件，不进入策略通信输出。
      *diagnostics_ << Json{{"round_id", round_},
                            {"sim_time_ns", now},
                            {"tracker_state", modules::TrackerStateName(input_.prediction.state)},
                            {"search", search},
                            {"command_valid", output.command.valid},
                            {"reject_reason", modules::FireRejectReasonName(output.reject_reason)},
                            {"failure",
                             modules::GimbalTrajectoryFailureReasonName(p.failure_reason)},
                            {"warm_start", modules::GimbalWarmStartActionName(p.warm_start_action)},
                            {"rebase_reason", d.solver_warm_start_reset_reason},
                            {"reference_points", p.reference.size()},
                            {"reference_yaw_error",
                             p.reference.empty()
                                 ? 0.0
                                 : std::remainder(p.reference.front().yaw - d.feedback.yaw,
                                                  2 * std::numbers::pi)},
                            {"solve_us", p.solve_time_us},
                            {"step_us", std::chrono::duration<double, std::micro>(
                                            std::chrono::steady_clock::now() - step_start)
                                            .count()},
                            {"primary_yaw", axis(p.primary_yaw_solver)},
                            {"primary_pitch", axis(p.primary_pitch_solver)},
                            {"retry_yaw", axis(p.retry_yaw_solver)},
                            {"retry_pitch", axis(p.retry_pitch_solver)}}
                           .dump()
                    << '\n';
      Require(static_cast<bool>(*diagnostics_), "diagnostic write failed");
    }
    return response;
  }

 private:
  YAML::Node Load(const std::string& name) const {
    return YAML::LoadFile(root_ + "/" + name + ".yaml");
  }
  std::string root_;
  std::ostream* diagnostics_{nullptr};  ///< 可选离线诊断输出，不作为控制器输入。
  std::uint64_t round_{0}, processed_{0};
  std::optional<std::uint64_t> last_tick_, last_frame_;
  std::optional<std::uint64_t> search_start_;
  double search_origin_yaw_{0.0};
  std::optional<std::pair<std::uint64_t, std::uint64_t>> evaluation_window_;
  bool training_{false};
  std::string policy_mode_{"rule"};
  std::uint64_t policy_token_{0};
  std::unique_ptr<modules::FireOnlyPolicyAdapter> fire_only_;
  std::unique_ptr<modules::ArmorPnp> pnp_;
  std::unique_ptr<modules::ArmorPredictor> predictor_;
  std::unique_ptr<modules::ControlSession> control_;
  modules::ControlInputSnapshot input_;
  std::optional<modules::ControlStepResult> pending_;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: rmvision-rl-bridge CONFIG_MODULE_DIR LOGGER_YAML [DIAGNOSTICS_JSONL]\n";
    return 2;
  }
  try {
    mv::Logger::Instance().InitFromFile(argv[2]);
    std::ofstream diagnostics;
    if (argc == 4) {
      diagnostics.open(argv[3]);
      Require(diagnostics.is_open(), "cannot open diagnostic output");
    }
    Bridge bridge(argv[1], argc == 4 ? &diagnostics : nullptr);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.size() > 1024 * 1024)
        throw std::runtime_error("bridge input too large");
      try {
        std::cout << bridge.Handle(Json::parse(line)).dump() << std::endl;
      } catch (const PolicyCancelled&) {
        bridge.Discard();
        std::cout << Json({{"ok", true}, {"kind", "cancelled"}}).dump() << std::endl;
      } catch (const std::exception& e) {
        // 部分更新的估计或控制状态不可复用，出错后结束进程，由调用方重新启动。
        std::cout << Json({{"ok", false}, {"error", e.what()}}).dump() << std::endl;
        return 1;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
