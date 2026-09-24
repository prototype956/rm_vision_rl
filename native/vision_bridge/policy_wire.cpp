#include "policy_wire.hpp"

#include <cmath>
#include <nlohmann/json.hpp>

namespace rmvision_rl {
using Json = nlohmann::json;
namespace {
Json Vector(const mv::geometry::Vector3& v) { return {v.x(), v.y(), v.z()}; }
Json Age(double value) { return std::isfinite(value) ? Json(value) : Json(nullptr); }

}  // namespace

Json Observation(const mv::modules::PolicyObservation& o) {
  const auto& e = o.estimate;
  const auto& f = o.feedback;
  const auto& r = o.referee;
  Json candidates = Json::array();
  for (std::size_t i = 0; i < o.candidates.size(); ++i) {
    const auto& c = o.candidates[i];
    candidates.push_back({{"valid", c.valid},
                          {"slot", c.slot},
                          {"target_world", Vector(c.target_world)},
                          {"yaw_rad", c.yaw},
                          {"pitch_rad", c.pitch},
                          {"distance_m", c.distance_m},
                          {"fly_time_s", c.fly_time_s},
                          {"facing_now_rad", o.facing_now_rad[i]},
                          {"facing_impact_rad", o.facing_impact_rad[i]},
                          {"prediction_horizon_s", c.prediction_horizon_s}});
  }
  Json covariance = Json::array();
  for (int i = 0; i < 3; ++i)
    covariance.push_back({e.center_covariance_world(i, 0), e.center_covariance_world(i, 1),
                          e.center_covariance_world(i, 2)});
  Json chassis = nullptr;
  if (o.chassis_motion) {
    const auto& c = *o.chassis_motion;
    chassis = {{"yaw_rad", c.yaw_rad},
               {"velocity_body_mps", {c.velocity_body_mps.x(), c.velocity_body_mps.y()}},
               {"yaw_velocity_rad_s", c.yaw_velocity_rad_s}};
  }
  return {{"wire_version", 1},
          {"observation_version", o.VERSION},
          {"action_version", 1},
          {"estimate",
           {{"state", mv::modules::TrackerStateName(e.state)},
            {"label", e.label ? Json(static_cast<int>(*e.label)) : Json(nullptr)},
            {"armor_type", e.type ? Json(static_cast<int>(*e.type)) : Json(nullptr)},
            {"state_vector", e.state_vector},
            {"covariance_diagonal", e.covariance_diagonal},
            {"center_world", Vector(e.center_world)},
            {"velocity_world", Vector(e.velocity_world)},
            {"orientation_xyzw",
             {e.orientation_world.x(), e.orientation_world.y(), e.orientation_world.z(),
              e.orientation_world.w()}},
            {"yaw_velocity_rad_s", e.yaw_velocity_rad_s},
            {"radii_m", e.radii_m},
            {"height_offset_m", e.height_offset_m},
            {"armor_tilt_rad", e.armor_tilt_rad},
            {"center_covariance_world", covariance},
            {"yaw_variance_rad2", e.yaw_variance_rad2}}},
          {"feedback",
           {{"valid", f.valid},
            {"yaw_rad", f.yaw},
            {"pitch_rad", f.pitch},
            {"yaw_velocity_rad_s", f.yaw_velocity},
            {"pitch_velocity_rad_s", f.pitch_velocity}}},
          {"referee",
           {{"valid", r.valid},
            {"alive", r.alive},
            {"fire_permitted", r.fire_permitted},
            {"unlimited", r.unlimited},
            {"allowance_remaining", r.allowance_remaining},
            {"fire_blocks", r.fire_blocks},
            {"heat", r.heat},
            {"heat_limit", r.heat_limit},
            {"cooling_per_second", r.cooling_per_second}}},
          {"chassis_motion", chassis},
          {"candidates", candidates},
          {"action_mask", o.action_mask},
          {"prediction_age_s", Age(o.prediction_age_s)},
          {"feedback_age_s", Age(o.feedback_age_s)},
          {"referee_age_s", Age(o.referee_age_s)},
          {"previous_slot", o.previous_slot},
          {"selected_slot_age_s", o.selected_slot_age_s},
          {"since_request_s", o.since_request_s ? Json(*o.since_request_s) : Json(nullptr)}};
}
}  // namespace rmvision_rl
