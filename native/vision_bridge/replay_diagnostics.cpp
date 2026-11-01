#include "replay_diagnostics.hpp"

#include <cstring>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "tool/foxglove/simulation/combat_message_encoder.hpp"
namespace rmvision_rl {
using Json = nlohmann::json;
using namespace mv;
namespace {
constexpr std::uint64_t EPOCH = 1'000'000'000'000'000'000ULL;
geometry::Vector3 Point(const Json& p) {
  return {p.at(0).get<double>(), p.at(1).get<double>(), p.at(2).get<double>()};
}
void FillRobot(simulation::RobotCombatMeta& r, const Json& source) {
  const auto& x = source.at("resources");
  r.robot_id = source.at("robot_id");
  r.hp = x.at("hp");
  r.heat = x.at("heat");
  r.heat_limit = x.at("heat_limit");
  r.cooling_per_second = x.at("cooling_per_second");
  r.life = x.at("life_status") == "Alive" ? 0 : 1;
  r.allowance_mode = x.at("allowance").at("mode") == "limited" ? 1 : 0;
  r.allowance_remaining = x.at("allowance").value("remaining", 0U);
  r.actual_shots = x.at("actual_shots");
  r.rejected_requests = x.at("rejected_requests");
  r.damage_dealt = x.at("damage_dealt");
  r.damage_taken = x.at("damage_taken");
  r.kills = x.at("kills");
  r.damaging_hits = x.at("damaging_hits");
  r.heat_lock_count = x.at("heat_lock_count");
  r.heat_locked_s = source.at("heat_locked_s");
}
}  // namespace
ReplayDiagnostics::ReplayDiagnostics(const std::string& path, Json info,
                                     modules::ArmorPnpConfig config)
    : recorder_(path), evaluator_({}, config), info_(std::move(info)) {
  info_["source"] = "synthetic_measurements";
  info_["availability"] = {{"detector", false},    {"lightbars", false}, {"corner_refiner", false},
                           {"real_camera", false}, {"realtime", false},  {"pnp", true},
                           {"prediction", true},   {"control", true}};
}
void ReplayDiagnostics::Frame(tool::foxglove::pipeline::VisionDebugFrame frame) {
  if (frames_.size() >= 32) throw std::runtime_error("unflushed replay frame overflow");
  frames_.push_back(std::move(frame));
}
void ReplayDiagnostics::Control(const modules::FireControlResult& result) {
  if (control_) throw std::runtime_error("unflushed replay control cycle");
  control_ = result;
  info_["control_source"] = result.decision ? "model" : "rule_or_search";
  info_["action"] = result.decision ? Json(result.decision->action) : Json(nullptr);
}
void ReplayDiagnostics::Snapshot(const Json& payload) {
  const auto& response = payload.at("response");
  const auto& data = response.at("data");
  last_ns_ = EPOCH + response.at("sim_time_ns").get<std::uint64_t>();
  info_["sim_time_ns"] = response.at("sim_time_ns");
  info_["phase"] = payload.value("phase", std::string("warming"));
  info_["complete"] = false;
  if (payload.contains("summary")) info_["summary"] = payload.at("summary");
  for (auto& f : frames_) {
    const auto sequence = f.packet.capture.stamp.sequence;
    const auto capture = *f.packet.capture.stamp.capture_timestamp_ns;
    const auto& truths = data.at("evaluation").at("visual_truth");
    const Json* truth = nullptr;
    const Json* original = nullptr;
    for (const auto& t : truths)
      if (t.at("sequence") == sequence && t.at("round_id") == response.at("round_id") &&
          EPOCH + t.at("capture_time_ns").get<std::uint64_t>() == capture)
        truth = &t;
    for (const auto& v : data.at("visual_frames"))
      if (v.at("sequence") == sequence) original = &v;
    if (!truth || !original) throw std::runtime_error("missing capture-aligned replay truth");
    // Asymmetric observations zero the platform translation before PnP. Apply exactly
    // that origin shift to display/evaluation truth, without changing controller inputs.
    const geometry::Vector3 origin =
        Point(original->at("kinematics").at("world_t_gimbal").at("translation")) -
        f.packet.kinematics->world_t_gimbal.translation;
    simulation::SimulationFrameData sim;
    for (const auto& p : truth->at("plates")) {
      simulation::GroundTruthArmor a;
      a.id = p.at("id");
      a.owner_robot_id = p.at("owner_robot_id");
      a.team = p.at("team");
      a.label = p.at("label");
      a.type = static_cast<geometry::ArmorType>(p.at("armor_type").get<int>());
      a.width_m = p.at("width_m");
      a.height_m = p.at("height_m");
      a.world_t_armor.translation = Point(p.at("center_world")) - origin;
      const auto& q = p.at("rotation_xyzw");
      a.world_t_armor.rotation = geometry::Quaternion(q.at(3).get<double>(), q.at(0).get<double>(),
                                                      q.at(1).get<double>(), q.at(2).get<double>());
      for (std::size_t i = 0; i < 4; ++i)
        a.corners_world[i] = Point(p.at("corners_world").at(i)) - origin;
      sim.armors.push_back(a);
    }
    for (const auto& t : truth->at("targets")) {
      simulation::GroundTruthTarget v;
      v.id = t.at("id");
      v.team = t.at("team");
      v.armor_label = t.at("armor_label");
      v.position_world = Point(t.at("position_world")) - origin;
      v.yaw = t.at("yaw");
      v.yaw_velocity = t.at("yaw_velocity");
      sim.targets.push_back(v);
    }
    f.packet.simulation = std::move(sim);
    const auto& camera = *f.packet.camera_model;
    f.packet.capture.image = cv::Mat(camera.height, camera.width, CV_8UC3, cv::Scalar(24, 24, 24));
    cv::putText(f.packet.capture.image, "SYNTHETIC MEASUREMENTS - NO CAMERA IMAGE", {15, 28},
                cv::FONT_HERSHEY_SIMPLEX, .6, {200, 200, 200}, 1);
    f.simulation_evaluation = evaluator_.Evaluate(
        {f.packet.capture.stamp, camera, *f.packet.kinematics, *f.packet.simulation,
         f.output.detections, f.output.refinements, f.diagnostics.refinements, f.output.pnp,
         f.diagnostics.pnp, f.output.prediction});
    std::optional<modules::ArmorImpactSnapshot> impact;
    if (control_ && control_->output.source_sequence == sequence) {
      impact = modules::ArmorImpactSnapshot{.source_sequence = sequence,
                                            .selected_slot = control_->output.selected_slot,
                                            .ballistic = control_->output.ballistic};
      const auto& o = control_->output;
      const auto& d = control_->diagnostics.armor_selection;
      f.armor_selection =
          modules::ArmorSelectionSnapshot{.valid = true,
                                          .source_sequence = sequence,
                                          .tracker_state = o.tracker_state,
                                          .tracked_label = o.tracked_label,
                                          .tracked_type = o.tracked_type,
                                          .selected_slot = o.selected_slot,
                                          .pending_slot = d.pending_slot,
                                          .pending_duration_s = d.pending_duration_s,
                                          .switch_confirmation_s = d.switch_confirmation_s};
    }
    recorder_.Vision(f, impact);
    ++vision_samples_;
  }
  frames_.clear();
  if (control_) {
    recorder_.Control(*control_);
    control_.reset();
    ++control_samples_;
  }
  simulation::CombatFrameMeta combat{};
  combat.round_id = response.at("round_id");
  combat.sim_time_ns = response.at("sim_time_ns");
  const auto& robots = data.at("evaluation").at("robots");
  if (robots.size() > simulation::COMBAT_MAX_ROBOTS)
    throw std::runtime_error("too many combat robots");
  combat.robot_count = static_cast<std::uint32_t>(robots.size());
  for (std::size_t i = 0; i < robots.size(); ++i) FillRobot(combat.robots[i], robots.at(i));
  const auto& referee = data.at("self_referee");
  combat.referee_valid = referee.at("valid");
  combat.referee_sample_ns = referee.at("sample_ns");
  combat.referee_sample_sequence = referee.at("sequence");
  auto& self = combat.self_referee;
  self.robot_id = 1;
  self.hp = referee.at("hp");
  self.life = self.hp > 0 ? 0 : 1;
  self.heat = referee.at("heat");
  self.heat_limit = referee.at("heat_limit");
  self.cooling_per_second = referee.at("cooling_per_second");
  self.allowance_mode = referee.at("allowance_mode");
  self.allowance_remaining = referee.at("allowance_remaining");
  self.fire_permitted = referee.at("fire_permitted");
  self.fire_blocks = referee.at("fire_blocks");
  for (const auto& e : data.at("events")) {
    if (combat.event_count == simulation::COMBAT_MAX_EVENTS)
      throw std::runtime_error("combat event overflow");
    auto& event = combat.events[combat.event_count++];
    event.id = e.at("event_id");
    event.round_id = e.at("round_id");
    event.round_time_ns = e.at("time_ns");
    const auto detail = e.at("kind").get<std::string>();
    std::memcpy(event.detail, detail.data(), std::min(detail.size(), sizeof(event.detail) - 1));
  }
  recorder_.Combat(combat, last_ns_);
  recorder_.Physical(Json{{"round_id", response.at("round_id")},
                          {"sim_time_ns", response.at("sim_time_ns")},
                          {"evaluation", data.at("evaluation")},
                          {"events", data.at("events")},
                          {"self_referee", data.at("self_referee")}}
                         .dump(),
                     last_ns_);
  recorder_.Info(info_.dump(), last_ns_);
}
void ReplayDiagnostics::Close(const Json& summary) {
  if (!frames_.empty() || control_) throw std::runtime_error("unflushed replay data at close");
  info_["phase"] = "complete";
  info_["complete"] = true;
  info_["summary"] = summary;
  info_["summary"]["control_samples"] = control_samples_;
  info_["summary"]["vision_samples"] = vision_samples_;
  recorder_.Info(info_.dump(), last_ns_);
  recorder_.Close();
}
}  // namespace rmvision_rl
