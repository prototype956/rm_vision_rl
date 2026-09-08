#include "modules/armor_predictor/armor_prediction_output.hpp"
#include "modules/armor_predictor/detail/armor_motion_model.hpp"
#include <stdexcept>
namespace mv::modules {
PredictionHorizon ExtrapolatePrediction(const ArmorPredictionOutput& prediction, double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0)
    throw std::invalid_argument("prediction horizon must be finite and nonnegative");
  detail::NominalState state;
  state.position_world = prediction.center_world;
  state.velocity_world = prediction.velocity_world;
  state.world_q_car = prediction.orientation_world;
  state.yaw_velocity_rad_s = prediction.yaw_velocity_rad_s;
  state.log_radius_1 = std::log(prediction.radii_m[0]);
  state.log_radius_2 = std::log(prediction.radii_m[1]);
  state.height_offset_m = prediction.height_offset_m;
  const auto FUTURE = detail::PredictState(state, seconds);
  PredictionHorizon horizon;
  horizon.seconds = seconds;
  horizon.center_world = FUTURE.position_world;
  horizon.orientation_world = FUTURE.world_q_car;
  horizon.yaw = detail::HeadingYaw(FUTURE);
  for (int slot = 0; slot < 4; ++slot) {
    horizon.armors[slot] = {.slot = slot,
                            .world_t_armor = detail::WorldArmorPose(
                                FUTURE, {.slot = slot, .tilt_rad = prediction.armor_tilt_rad})};
  }
  return horizon;
}

}  // namespace mv::modules
